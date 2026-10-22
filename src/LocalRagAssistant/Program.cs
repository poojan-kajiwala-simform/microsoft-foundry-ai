using System.Text;
using System.Text.Json;
using LocalRag.Core;
using LocalRag.Core.Extraction;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using LocalRag.Core.Ingestion;
using LocalRag.Core.Storage;
using LocalRag.Foundry;
using LocalRagAssistant;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddLocalRagCore(builder.Configuration);
builder.Services.AddFoundryLocal(); // <- the only provider-specific line; swap for Azure OpenAI/Ollama by registering IChatClient + IEmbeddingGenerator instead

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(o => o.SwaggerDoc("v1", new() { Title = "Local RAG Assistant", Version = "v1" }));
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.Converters.Add(new System.Text.Json.Serialization.JsonStringEnumConverter()));
builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = 100 * 1024 * 1024);

var app = builder.Build();
var json = new JsonSerializerOptions(JsonSerializerDefaults.Web);

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}
app.UseDefaultFiles();
app.UseStaticFiles();

string DocumentsFolder() =>
    Path.GetFullPath(app.Services.GetRequiredService<IOptions<RagOptions>>().Value.DocumentsFolder, app.Environment.ContentRootPath);

app.MapGet("/health", (FoundryLocalService foundry, VectorIndex index, IOptions<RagOptions> o) => Results.Ok(new
{
    runtimeRunning = foundry.RuntimeRunning,
    chat = foundry.Chat,
    embedding = foundry.Embedding,
    indexedChunks = index.Count,
    minSimilarity = o.Value.MinSimilarity,
}))
.WithSummary("Runtime state, per-model lifecycle (download/load timings), and index size.");

app.MapGet("/documents", async (SqliteChunkRepository repo) => Results.Ok(await repo.GetDocumentsAsync()))
.WithSummary("Lists indexed documents.");

app.MapDelete("/documents", async (string source, IngestionPipeline ingestion, CancellationToken ct) =>
    await ingestion.RemoveAsync(source, ct) ? Results.NoContent() : Results.NotFound())
.WithSummary("Removes one document (by its relative path) from the index.");

// POST /ingest            -> sync the configured documents folder (add / update / skip unchanged / remove deleted)
// POST /ingest + files    -> save uploads into that folder, then sync
app.MapPost("/ingest", async (HttpRequest request, IngestionPipeline ingestion, CancellationToken ct) =>
{
    var folder = DocumentsFolder();
    Directory.CreateDirectory(folder);

    if (request.HasFormContentType)
    {
        var form = await request.ReadFormAsync(ct);
        foreach (var file in form.Files)
        {
            var name = Path.GetFileName(file.FileName); // strips any path segments: no traversal out of the folder
            if (!TextExtractor.IsSupported(name))
                return Results.BadRequest(new { error = $"Unsupported file type: {name}. Supported: .pdf .docx .txt .md" });
            await using var target = File.Create(Path.Combine(folder, name));
            await file.CopyToAsync(target, ct);
        }
    }

    try
    {
        return Results.Ok(await ingestion.SyncFolderAsync(folder, ct));
    }
    catch (IngestionBusyException ex) { return Results.Problem(ex.Message, statusCode: 409); }
    catch (ModelNotReadyException ex) { return Results.Problem(ex.Message, statusCode: 503); }
})
.DisableAntiforgery()
.WithSummary("Syncs the documents folder into the index (idempotent). Optionally upload files as multipart/form-data first.");

// Returns { answer } once generation has finished (no streaming).
app.MapPost("/ask", async (AskRequest req, RagAnswerService rag, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(req.Query)) return Results.BadRequest(new { error = "query is required." });

    try
    {
        await foreach (var e in rag.AskAsync(req.Query, null, ct))
            if (e is DoneEvent d) return Results.Ok(new { answer = d.Answer });
        return Results.Problem("No answer was produced.", statusCode: 500);
    }
    catch (ModelNotReadyException ex) { return Results.Problem(ex.Message, statusCode: 503); }
    catch (ReindexRequiredException ex) { return Results.Problem(ex.Message, statusCode: 409); }
})
.WithSummary("Answers a question from the indexed documents and returns only the final answer.");

app.Run();

namespace LocalRagAssistant
{
    public sealed record AskRequest(string Query);
}
