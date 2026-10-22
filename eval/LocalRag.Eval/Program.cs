using System.Diagnostics;
using System.Text;
using System.Text.Json;
using LocalRag.Core;
using LocalRag.Core.Chunking;
using LocalRag.Core.Extraction;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using LocalRag.Core.Ingestion;
using LocalRag.Core.Retrieval;
using LocalRag.Core.Storage;
using LocalRag.Eval;
using LocalRag.Foundry;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

// Usage: dotnet run --project eval/LocalRag.Eval -c Release -- [--corpus eval/corpus] [--dataset eval/golden.jsonl]
//          [--generate] [--judge] [--label cpu-laptop] [--Rag:RegisterExecutionProviders=true] [--Rag:TopK=3] ...
// Any --Rag:Key=Value argument overrides the app's configuration (model aliases, chunk size, threshold, ...).

var arg = (string name, string fallback) =>
{
    var i = Array.IndexOf(args, "--" + name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
};
var flag = (string name) => args.Contains("--" + name);

var repoRoot = FindRepoRoot();
var corpus = Path.GetFullPath(arg("corpus", Path.Combine(repoRoot, "eval", "corpus")));
var datasetPath = Path.GetFullPath(arg("dataset", Path.Combine(repoRoot, "eval", "golden.jsonl")));
var label = arg("label", Environment.MachineName);
var generate = flag("generate");

var judge = flag("judge");

var builder = Host.CreateApplicationBuilder(args);
builder.Configuration.AddJsonFile(Path.Combine(repoRoot, "src", "LocalRagAssistant", "appsettings.json"), optional: true);
builder.Configuration.AddCommandLine(args); // re-add so --Rag:* wins over the json file
builder.Logging.ClearProviders().AddSimpleConsole(o => o.SingleLine = true);
builder.Logging.SetMinimumLevel(LogLevel.Warning).AddFilter("LocalRag", LogLevel.Information);
builder.Services.Configure<RagOptions>(builder.Configuration.GetSection(RagOptions.SectionName));
builder.Services.AddFoundryLocal();
using var host = builder.Build();

var report = new StringBuilder();
void Say(string line = "") { Console.WriteLine(line); report.AppendLine(line); }

var baseOptions = host.Services.GetRequiredService<IOptions<RagOptions>>().Value;
var foundry = host.Services.GetRequiredService<FoundryLocalService>();
var embedder = host.Services.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
var chat = host.Services.GetRequiredService<IChatClient>();

Console.WriteLine("Starting Foundry Local and waiting for models (first run downloads them)...");
await host.StartAsync();
var chatAvailable = baseOptions.LoadChatModel;
string? chatProblem = null;
try { await foundry.WaitUntilReadyAsync(); }
catch (ModelNotReadyException ex) when (baseOptions.LoadChatModel && foundry.Embedding.State == ModelLoadState.Ready)
{
    // The chat model is the heavy one; a failed load (typically out of memory) must not hide the retrieval results.
    chatAvailable = false;
    chatProblem = ex.Message;
}

var golden = File.ReadAllLines(datasetPath).Where(l => l.Trim().Length > 0)
    .Select(l => JsonSerializer.Deserialize<GoldenQuestion>(l, new JsonSerializerOptions(JsonSerializerDefaults.Web))!).ToList();

Say($"# Local RAG evaluation: {label}");
Say();
Say($"- Date: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC");
Say($"- Machine: {Environment.OSVersion.VersionString}, {Environment.ProcessorCount} logical cores, {Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER")}, {GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 1024 / 1024 / 1024.0:F0} GB RAM");
var gpu = Hardware.QueryNvidia();
if (gpu is not null) Say($"- GPU: {gpu}");
Say($"- Execution providers registered: {baseOptions.RegisterExecutionProviders}");
Say($"- Chat model: {(chatAvailable ? $"{foundry.Chat.Alias} (variant `{foundry.Chat.VariantId}`)" : (chatProblem is null ? "not loaded" : $"FAILED to load: {chatProblem}"))}, embedding model: {foundry.Embedding.Alias} (variant `{foundry.Embedding.VariantId}`)");
Say($"- Cold start (this run): embedding load {foundry.Embedding.LoadTime?.TotalSeconds:F1}s" +
    (chatAvailable ? $", chat load {foundry.Chat.LoadTime?.TotalSeconds:F1}s" : "") +
    (foundry.Chat.DownloadTime is null ? " (models were already cached)" : $", chat download {foundry.Chat.DownloadTime?.TotalSeconds:F0}s"));
Say($"- Corpus: {Directory.GetFiles(corpus).Length} files; golden set: {golden.Count(g => g.Answerable)} answerable + {golden.Count(g => !g.Answerable)} unanswerable questions");
Say($"- Config: TopK={baseOptions.TopK}, MinSimilarity={baseOptions.MinSimilarity}, chunk={baseOptions.ChunkSizeWords}w/{baseOptions.ChunkOverlapWords}w overlap, context budget={baseOptions.MaxContextTokens} tokens");
Say();

// ---------- Variants: chunking strategy x hybrid ----------
var variants = new[]
{
    ("fixed 300w + vector", ChunkingStrategy.Fixed, false, baseOptions.ChunkSizeWords),
    ("fixed 60w + vector", ChunkingStrategy.Fixed, false, 60),
    ("structured 300w + vector", ChunkingStrategy.Structured, false, baseOptions.ChunkSizeWords),
    ("structured 300w + hybrid", ChunkingStrategy.Structured, true, baseOptions.ChunkSizeWords),
};

var stacks = new Dictionary<string, Stack>();
Say("## Ingestion");
Say();
Say("| Variant | Files | Chunks | Total s | Embedding s | Chunks/s | Re-run (no changes) s |");
Say("|---|---|---|---|---|---|---|");
foreach (var (name, strategy, hybrid, size) in variants)
{
    var options = Clone(baseOptions);
    options.Chunking = strategy;
    options.HybridSearch = hybrid;
    options.ChunkSizeWords = size;
    options.ChunkOverlapWords = Math.Min(baseOptions.ChunkOverlapWords, size / 4);
    options.DatabasePath = Path.Combine(Path.GetTempPath(), $"localrag-eval-{Guid.NewGuid():N}.db");
    var stack = new Stack(options, embedder, chat);
    var first = await stack.Ingestion.SyncFolderAsync(corpus);
    var second = await stack.Ingestion.SyncFolderAsync(corpus);
    foreach (var f in first.Failures) Say($"> ingest failure in {name}: {f.Source}: {f.Reason}");
    Say($"| {name} | {first.Added} | {first.Chunks} | {first.Elapsed.TotalSeconds:F1} | {first.EmbeddingTime.TotalSeconds:F1} | {first.Chunks / Math.Max(first.Elapsed.TotalSeconds, 0.001):F1} | {second.Elapsed.TotalSeconds:F2} (unchanged: {second.Unchanged}) |");
    stacks[name] = stack;
    if (first.Added == 0)
    {
        Say();
        Say($"**Aborting: nothing could be ingested for '{name}'; retrieval metrics would be meaningless.** On a memory-starved machine the chat and embedding models cannot both run; retry with --Rag:LoadChatModel=false or free memory.");
        File.WriteAllText(Path.Combine(Directory.CreateDirectory(Path.Combine(repoRoot, "eval", "results")).FullName, $"{label}-FAILED.md"), report.ToString());
        Environment.Exit(1);
    }
}
Say();
Say($"Process memory after ingestion: working set {Process.GetCurrentProcess().WorkingSet64 / 1024 / 1024} MB (models run in-process).");
Say();

// ---------- Retrieval ----------
Say("## Retrieval quality");
Say();
Say($"Hit@K = a chunk from the expected source containing the evidence phrase is in the top {baseOptions.TopK}; Hit@1 requires it to be ranked first (the stricter test on a corpus this small). MRR uses the first such chunk. Margin > 0 means a single similarity threshold can separate answerable from unanswerable questions.");
Say();
Say("| Variant | Hit@1 | Hit@K | MRR | Margin (min answerable − max unanswerable) | Answerable: top similarity min / median | Unanswerable: top similarity median / max | Abstention accuracy @ threshold | Best threshold on this set |");
Say("|---|---|---|---|---|---|---|---|---|");
var scoresByVariant = new Dictionary<string, List<(GoldenQuestion Q, RetrievalResult R)>>();
foreach (var (name, _, _, _) in variants)
{
    var stack = stacks[name];
    var results = new List<(GoldenQuestion, RetrievalResult)>();
    foreach (var q in golden) results.Add((q, await stack.Retriever.RetrieveAsync(q.Question)));
    scoresByVariant[name] = results;

    var answerable = results.Where(r => r.Item1.Answerable).ToList();
    var unanswerable = results.Where(r => !r.Item1.Answerable).ToList();
    int hits = 0, hit1 = 0; double mrr = 0;
    foreach (var (q, r) in answerable)
    {
        var rank = r.Chunks.Select((c, i) => (c, i)).FirstOrDefault(x => Matches(x.c.Chunk, q));
        if (rank.c is not null) { hits++; if (rank.i == 0) hit1++; mrr += 1.0 / (rank.i + 1); }
    }
    var aTop = answerable.Select(r => r.Item2.TopSimilarity).OrderBy(x => x).ToList();
    var uTop = unanswerable.Select(r => r.Item2.TopSimilarity).OrderBy(x => x).ToList();
    var correct = answerable.Count(r => r.Item2.Confident) + unanswerable.Count(r => !r.Item2.Confident);
    var best = BestThreshold(aTop, uTop);
    Say($"| {name} | {hit1}/{answerable.Count} | {hits}/{answerable.Count} ({100.0 * hits / answerable.Count:F0}%) | {mrr / answerable.Count:F2} | {aTop.First() - uTop.Last():+0.00;-0.00} | {aTop.First():F2} / {Median(aTop):F2} | {Median(uTop):F2} / {uTop.Last():F2} | {correct}/{results.Count} | {best:F2} |");
}
Say();
Say("Misses (best variant):");
foreach (var (q, r) in scoresByVariant["structured 300w + hybrid"].Where(x => x.Q.Answerable))
{
    if (!r.Chunks.Any(c => Matches(c.Chunk, q)))
        Say($"- {q.Id} \"{q.Question}\" -> got {string.Join(", ", r.Chunks.Select(c => $"{c.Chunk.Source} ({c.Similarity:F2})"))}");
}
Say();

// ---------- Generation ----------
if (generate && !chatAvailable) Say("> `--generate` skipped: the chat model is not loaded (see the Chat model line above).");
else if (generate)
{
    var stack = stacks["structured 300w + hybrid"];
    var answers = stack.Answers();
    Say("## Generation (structured + hybrid)");
    Say();
    Say("| Id | Answerable | Declined | Retrieval ms | TTFT ms | Total s | Fragments/s | Keywords OK | Cited expected source | Grounded (heuristic) |" + (judge ? " Groundedness (LLM judge 1-5) |" : ""));
    Say("|---|---|---|---|---|---|---|---|---|---|" + (judge ? "---|" : ""));

    var rows = new List<GenRow>();
    foreach (var q in golden)
    {
        var row = await RunOne(answers, q, stack, judge);
        rows.Add(row);
        Say($"| {q.Id} | {(q.Answerable ? "yes" : "no")} | {row.Declined} | {row.RetrievalMs:F0} | {row.TtftMs?.ToString("F0") ?? "-"} | {row.TotalS:F1} | {row.FragmentsPerSecond?.ToString("F1") ?? "-"} | {row.KeywordsOk?.ToString() ?? "-"} | {row.CitedExpected?.ToString() ?? "-"} | {row.Grounded?.ToString() ?? "-"} |" + (judge ? $" {row.JudgeScore?.ToString("F0") ?? "-"} |" : ""));
    }

    var ans = rows.Where(r => r.Q.Answerable).ToList();
    var unans = rows.Where(r => !r.Q.Answerable).ToList();
    Say();
    Say("| Metric | Value |");
    Say("|---|---|");
    Say($"| Answerable: keyword-correct | {ans.Count(r => r.KeywordsOk == true)}/{ans.Count} |");
    Say($"| Answerable: cites the expected source | {ans.Count(r => r.CitedExpected == true)}/{ans.Count} |");
    Say($"| Answerable: wrongly declined | {ans.Count(r => r.Declined != "-")}/{ans.Count} |");
    Say($"| Unanswerable: correctly declined | {unans.Count(r => r.Declined != "-")}/{unans.Count} |");
    Say($"| Grounded (heuristic) among non-declined | {rows.Count(r => r.Grounded == true)}/{rows.Count(r => r.Grounded is not null)} |");
    if (judge) Say($"| Mean LLM-judge groundedness (1-5; judge is the same small local model: indicative only) | {rows.Where(r => r.JudgeScore is not null).Select(r => r.JudgeScore!.Value).DefaultIfEmpty(double.NaN).Average():F2} |");
    var ttft = rows.Where(r => r.TtftMs is not null).Select(r => r.TtftMs!.Value).OrderBy(x => x).ToList();
    var fps = rows.Where(r => r.FragmentsPerSecond is not null).Select(r => r.FragmentsPerSecond!.Value).OrderBy(x => x).ToList();
    if (ttft.Count > 0) Say($"| Time to first token (generated answers) | median {Median(ttft):F0} ms, p95 {ttft[(int)Math.Min(ttft.Count - 1, Math.Ceiling(ttft.Count * 0.95) - 1)]:F0} ms |");
    if (fps.Count > 0) Say($"| Streaming speed (fragments ≈ tokens per second) | median {Median(fps):F1} |");
    var p = Process.GetCurrentProcess();
    Say($"| Process memory (working set / peak) | {p.WorkingSet64 / 1024 / 1024} MB / {p.PeakWorkingSet64 / 1024 / 1024} MB |");
    if (Hardware.QueryNvidia() is { } gpuAfter) Say($"| GPU after run | {gpuAfter} |");
    Say();
    Say("Answers:");
    foreach (var r in rows) Say($"- **{r.Q.Id}** {r.Q.Question}\n  > {r.Answer.Replace("\n", " ")}");
}

var outDir = Path.Combine(repoRoot, "eval", "results");
Directory.CreateDirectory(outDir);
var outFile = Path.Combine(outDir, $"{label}-{DateTime.UtcNow:yyyyMMdd-HHmm}.md");
File.WriteAllText(outFile, report.ToString());
Console.WriteLine($"\nWrote {outFile}");

await host.StopAsync();
Environment.Exit(0); // Foundry's native runtime can keep the process alive after shutdown.

// ================= helpers =================
static bool Matches(ChunkRecord chunk, GoldenQuestion q) =>
    chunk.Source == q.ExpectedSource && (q.Evidence is null || chunk.Content.Contains(q.Evidence, StringComparison.OrdinalIgnoreCase));

static double Median(List<double> sorted) => sorted.Count == 0 ? double.NaN : sorted[sorted.Count / 2];

static double BestThreshold(List<double> answerableTop, List<double> unanswerableTop)
{
    var candidates = answerableTop.Concat(unanswerableTop).Distinct().OrderBy(x => x).ToList();
    return candidates.Select(t => (t, ok: answerableTop.Count(a => a >= t) + unanswerableTop.Count(u => u < t)))
        .OrderByDescending(x => x.ok).ThenByDescending(x => x.t).First().t;
}

static RagOptions Clone(RagOptions o) => JsonSerializer.Deserialize<RagOptions>(JsonSerializer.Serialize(o))!;

static string FindRepoRoot()
{
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "LocalRag.sln"))) dir = dir.Parent;
    return dir?.FullName ?? Directory.GetCurrentDirectory();
}

static async Task<GenRow> RunOne(RagAnswerService answers, GoldenQuestion q, Stack stack, bool judge)
{
    DoneEvent? done = null;
    await foreach (var e in answers.AskAsync(q.Question))
        if (e is DoneEvent d) done = d;
    if (done is null) throw new InvalidOperationException("No terminal event.");

    var declined = done.AbstainReason ?? "-";
    double? fps = null;
    if (done.Timings.FirstToken is { } ft && done.Timings.StreamedFragments > 1)
    {
        var span = (done.Timings.Total - ft).TotalSeconds;
        if (span > 0) fps = (done.Timings.StreamedFragments - 1) / span;
    }

    bool? keywordsOk = null, citedExpected = null, grounded = null;
    double? judgeScore = null;
    if (q.Answerable)
    {
        keywordsOk = q.AnswerKeywords is null || q.AnswerKeywords.Any(k => done.Answer.Contains(k, StringComparison.OrdinalIgnoreCase));
        citedExpected = done.CitedIndexes.Any(i => done.Retrieved.FirstOrDefault(c => c.Index == i)?.Source == q.ExpectedSource);
    }
    if (done.AbstainReason is null)
    {
        grounded = Groundedness.Heuristic(done.Answer, done.Retrieved);
        if (judge) judgeScore = await Groundedness.JudgeAsync(stack.Chat, q.Question, done.Answer, string.Join(' ', done.Retrieved.Select(c => c.Text)));
    }

    return new GenRow(q, done.Answer, declined, done.Timings.Retrieval.TotalMilliseconds, done.Timings.FirstToken?.TotalMilliseconds,
        done.Timings.Total.TotalSeconds, fps, keywordsOk, citedExpected, grounded, judgeScore);
}
