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

builder.Services.AddProblemDetails(); // unexpected failures (out of memory in a model, SQLite I/O) become application/problem+json
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(o => o.SwaggerDoc("v1", new() { Title = "Local RAG Assistant", Version = "v1" }));
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.Converters.Add(new System.Text.Json.Serialization.JsonStringEnumConverter()));
builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = 100 * 1024 * 1024);

var app = builder.Build();
var json = new JsonSerializerOptions(JsonSerializerDefaults.Web);
const int MaxQueryLength = 2000;

app.UseExceptionHandler();
app.UseStatusCodePages();
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
// Multipart is a CORS "simple" request, so any web page could POST here without a preflight. A custom header forces a
// preflight that cross-site pages cannot pass (this app sends no CORS headers), and uploads never overwrite silently.
app.MapPost("/ingest", async (HttpRequest request, bool? overwrite, IngestionPipeline ingestion, IOptions<RagOptions> ragOptions, CancellationToken ct) =>
{
    if (!request.Headers.ContainsKey(ClientHeader.Name))
        return Results.Problem($"Missing '{ClientHeader.Name}' header.", statusCode: 403);

    var folder = DocumentsFolder();
    Directory.CreateDirectory(folder);

    if (request.HasFormContentType)
    {
        var form = await request.ReadFormAsync(ct);

        // Validate every file before writing any, so a rejected upload leaves nothing behind.
        var uploads = new List<(IFormFile File, string Name)>();
        foreach (var file in form.Files)
        {
            var name = Path.GetFileName(file.FileName); // strips any path segments: no traversal out of the folder
            if (!TextExtractor.IsSupported(name))
                return Results.Problem($"Unsupported file type: {name}. Supported: .pdf .docx .txt .md", statusCode: 400);
            if (file.Length > ragOptions.Value.MaxFileBytes)
                return Results.Problem($"{name} is larger than the {ragOptions.Value.MaxFileBytes / (1024 * 1024)} MB limit.", statusCode: 400);
            if (overwrite != true && File.Exists(Path.Combine(folder, name)))
                return Results.Problem($"{name} already exists. Re-send with ?overwrite=true to replace it.", statusCode: 409,
                    extensions: new Dictionary<string, object?> { ["exists"] = name });
            uploads.Add((file, name));
        }

        foreach (var (file, name) in uploads)
        {
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

// Default: JSON with the same fields as the SSE "done" event. With ?stream=true, SSE events: sources -> token* -> done | error.
// Sources are announced once generation starts, so an abstention (no model call) is a single "done" event.
static object DonePayload(DoneEvent d) => new
{
    result = d.Answer,
    citations = d.Retrieved,
    citedIndexes = d.CitedIndexes,
    abstainReason = d.AbstainReason,
    timings = new
    {
        retrievalMs = (long)d.Timings.Retrieval.TotalMilliseconds,
        firstTokenMs = d.Timings.FirstToken is { } f ? (long?)f.TotalMilliseconds : null,
        totalMs = (long)d.Timings.Total.TotalMilliseconds,
        streamedFragments = d.Timings.StreamedFragments,
    },
};

app.MapPost("/ask", async (AskRequest req, bool? stream, HttpContext http, RagAnswerService rag, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(req.Query)) return Results.Problem("query is required.", statusCode: 400);
    if (req.Query.Length > MaxQueryLength) return Results.Problem($"query is limited to {MaxQueryLength} characters.", statusCode: 400);

    if (stream != true)
    {
        try
        {
            DoneEvent? done = null;
            await foreach (var e in rag.AskAsync(req.Query, null, ct))
                if (e is DoneEvent d) done = d;
            return Results.Ok(done is null ? new { result = (string?)null } : DonePayload(done));
        }
        catch (ModelNotReadyException ex) { return Results.Problem(ex.Message, statusCode: 503); }
        catch (ReindexRequiredException ex) { return Results.Problem(ex.Message, statusCode: 409); }
    }

    await using var events = rag.AskAsync(req.Query, null, ct).GetAsyncEnumerator(ct);
    var words = new WordBuffer();
    var started = false;

    async Task Write(string name, string data)
    {
        if (!started)
        {
            http.Response.ContentType = "text/event-stream";
            http.Response.Headers.CacheControl = "no-cache";
            started = true;
        }
        await http.Response.WriteAsync($"event: {name}\n", ct);
        foreach (var line in data.Split('\n')) await http.Response.WriteAsync($"data: {line}\n", ct);
        await http.Response.WriteAsync("\n", ct);
        await http.Response.Body.FlushAsync(ct);
    }

    try
    {
        while (await events.MoveNextAsync())
        {
            switch (events.Current)
            {
                case RetrievedEvent r:
                    await Write("sources", JsonSerializer.Serialize(r.Citations, json));
                    break;
                case TokenEvent t:
                    // Models stream sub-word fragments; send whole words so clients get readable text.
                    if (words.Push(t.Text) is { } word) await Write("token", word);
                    break;
                case DoneEvent d:
                    if (words.Flush() is { } rest) await Write("token", rest);
                    await Write("done", JsonSerializer.Serialize(DonePayload(d), json));
                    break;
            }
        }
    }
    catch (OperationCanceledException) { /* client went away or pressed Stop */ }
    // Filter only the exceptions we map, so anything else reaches UseExceptionHandler with its original stack trace.
    catch (ModelNotReadyException ex) when (!started) { return Results.Problem(ex.Message, statusCode: 503); }
    catch (ReindexRequiredException ex) when (!started) { return Results.Problem(ex.Message, statusCode: 409); }
    catch (Exception ex)
    {
        await Write("error", ex.Message);
    }
    return Results.Empty;
})
.WithSummary("Returns a grounded, cited answer as JSON; pass ?stream=true for Server-Sent Events (sources, token*, done).");

app.Run();

namespace LocalRagAssistant
{
    internal static class ClientHeader { public const string Name = "X-LocalRag-Client"; }

    public sealed record AskRequest(string Query);

    /// <summary>Accumulates streamed fragments and releases them at word boundaries.</summary>
    internal sealed class WordBuffer
    {
        private readonly StringBuilder _buffer = new();

        public string? Push(string fragment)
        {
            _buffer.Append(fragment);
            var text = _buffer.ToString();
            var cut = text.LastIndexOfAny([' ', '\n']);
            if (cut < 0) return null;
            _buffer.Clear().Append(text, cut + 1, text.Length - cut - 1);
            return text[..(cut + 1)];
        }

        public string? Flush()
        {
            if (_buffer.Length == 0) return null;
            var rest = _buffer.ToString();
            _buffer.Clear();
            return rest;
        }
    }
}
