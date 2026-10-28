using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using LocalRag.Core;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace LocalRag.Foundry;

public enum ModelLoadState { NotStarted, Downloading, Loading, Ready, Failed }

public sealed record ModelStatus(
    string Alias,
    string? VariantId,
    ModelLoadState State,
    double? DownloadPercent,
    TimeSpan? DownloadTime,
    TimeSpan? LoadTime,
    string? Error);

/// <summary>
/// Owns the Foundry Local lifecycle: start the runtime, then (in the background, so a multi-GB first-run download
/// never blocks the host) check the cache → download → load each model, and unload on shutdown.
/// The endpoint/port is managed by the SDK; nothing here hardcodes one.
/// </summary>
public sealed class FoundryLocalService : IHostedService
{
    private sealed class Slot(string alias)
    {
        public string Alias { get; } = alias;
        public string? VariantId;
        public volatile ModelLoadState State = ModelLoadState.NotStarted;
        public double? Percent;
        public TimeSpan? DownloadTime, LoadTime;
        public string? Error;
    }

    private readonly RagOptions _options;
    private readonly ILogger<FoundryLocalService> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private readonly Slot _chat, _embedding;
    private CancellationTokenSource? _cts;
    private Task _loadTask = Task.CompletedTask;

    // Session API (SDK 2.x). The embeddings session is stateless and safe to share. Chat sessions accumulate turn
    // history, so the chat adapter creates one per request from this loaded model instead of sharing a session.
    internal IModel? ChatModel { get; private set; }
    internal EmbeddingsSession? Embeddings { get; private set; }

    public FoundryLocalService(IOptions<RagOptions> options, ILogger<FoundryLocalService> logger, ILoggerFactory loggerFactory)
    {
        _options = options.Value;
        _logger = logger;
        _loggerFactory = loggerFactory;
        _chat = new Slot(_options.ChatModelAlias);
        _embedding = new Slot(_options.EmbeddingModelAlias);
    }

    public bool RuntimeRunning => FoundryLocalManager.IsInitialized;
    public ModelStatus Chat => Snapshot(_chat);
    public ModelStatus Embedding => Snapshot(_embedding);

    private static ModelStatus Snapshot(Slot s) => new(s.Alias, s.VariantId, s.State, s.Percent, s.DownloadTime, s.LoadTime, s.Error);

    internal void EnsureReady(bool chat)
    {
        var slot = chat ? _chat : _embedding;
        var ready = chat ? ChatModel is not null : Embeddings is not null;
        if (slot.State != ModelLoadState.Ready || !ready) throw new ModelNotReadyException(slot.Alias, slot.State.ToString(), slot.Error);
    }

    /// <summary>Completes when both models are Ready (or one failed). Used by CLI tools that must wait; the web host just polls /health.</summary>
    public async Task WaitUntilReadyAsync(CancellationToken ct = default)
    {
        while (true)
        {
            ct.ThrowIfCancellationRequested();
            if (_options.LoadChatModel && _chat.State == ModelLoadState.Failed) throw new ModelNotReadyException(_chat.Alias, "Failed", _chat.Error);
            if (_embedding.State == ModelLoadState.Failed) throw new ModelNotReadyException(_embedding.Alias, "Failed", _embedding.Error);
            if ((!_options.LoadChatModel || _chat.State == ModelLoadState.Ready) && _embedding.State == ModelLoadState.Ready) return;
            await Task.Delay(250, ct);
        }
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _cts = new CancellationTokenSource();
        await FoundryLocalManager.CreateAsync(new Configuration { AppName = _options.AppName }, _loggerFactory.CreateLogger("FoundryLocal"), cancellationToken);
        _logger.LogInformation("Foundry Local runtime started.");
        _loadTask = Task.Run(() => LoadAllAsync(_cts.Token)); // observed in StopAsync; LoadAllAsync handles its own failures
    }

    private async Task LoadAllAsync(CancellationToken ct)
    {
        try
        {
            if (_options.RegisterExecutionProviders)
            {
                // Pulls GPU/NPU execution providers (extra download). Failure just means CPU fallback.
                try { await FoundryLocalManager.Instance.DownloadAndRegisterEpsAsync(ct: ct); _logger.LogInformation("Execution providers registered."); }
                catch (Exception ex) when (ex is not OperationCanceledException) { _logger.LogWarning(ex, "Execution provider registration failed; continuing with CPU."); }
            }

            var catalog = await FoundryLocalManager.Instance.GetCatalogAsync(ct);
            // Embedding first: it is small, and ingestion/retrieval only need it.
            await LoadAsync(catalog, _embedding, isChat: false, ct);
            if (_options.LoadChatModel) await LoadAsync(catalog, _chat, isChat: true, ct);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogError(ex, "Unexpected error while initialising Foundry Local models.");
        }
    }

    private async Task LoadAsync(ICatalog catalog, Slot slot, bool isChat, CancellationToken ct)
    {
        try
        {
            slot.State = ModelLoadState.Downloading;
            var model = await catalog.GetModelAsync(slot.Alias, ct)
                ?? throw new InvalidOperationException($"Model alias '{slot.Alias}' was not found in the Foundry Local catalog (run `foundry model list`).");
            slot.VariantId = model.Id;

            if (!await model.IsCachedAsync(ct))
            {
                _logger.LogInformation("Downloading '{Alias}' ({Variant}); first run only...", slot.Alias, model.Id);
                var sw = Stopwatch.StartNew();
                var lastLogged = -10.0;
                await model.DownloadAsync(p =>
                {
                    slot.Percent = p;
                    if (p - lastLogged >= 10) { lastLogged = p; _logger.LogInformation("'{Alias}' download: {Percent:F0}%", slot.Alias, p); }
                }, ct);
                slot.DownloadTime = sw.Elapsed;
            }

            if (isChat) await CapSequenceLengthAsync(model, ct);

            slot.State = ModelLoadState.Loading;
            var load = Stopwatch.StartNew();
            await model.LoadAsync(ct);
            slot.LoadTime = load.Elapsed; // cold-start cost: reported by /health

            if (isChat) ChatModel = model;
            else Embeddings = new EmbeddingsSession(model);

            slot.State = ModelLoadState.Ready;
            _logger.LogInformation("'{Alias}' ready (variant {Variant}, load {Seconds:F1}s).", slot.Alias, slot.VariantId, slot.LoadTime.Value.TotalSeconds);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            slot.State = ModelLoadState.Failed;
            slot.Error = ex.Message;
            _logger.LogError(ex, "Failed to load model '{Alias}'.", slot.Alias);
        }
    }

    /// <summary>
    /// The runtime sizes the KV cache from genai_config.json's search.max_length, which long-context models (phi-3.5-mini:
    /// 131072) set to their full window — tens of GB, so loading fails. The request API cannot override it, so cap it in
    /// the cached config before loading. Idempotent; only ever lowers the value.
    /// </summary>
    private async Task CapSequenceLengthAsync(IModel model, CancellationToken ct)
    {
        try
        {
            var path = Path.Combine(await model.GetPathAsync(ct), "genai_config.json");
            if (!File.Exists(path)) return;

            var root = JsonNode.Parse(await File.ReadAllTextAsync(path, ct))!;
            var cap = _options.MaxSequenceLength;
            // The runtime may size the cache from either field, so lower both.
            var fields = new[] { (Obj: root["search"]?.AsObject(), Key: "max_length"), (Obj: root["model"]?.AsObject(), Key: "context_length") };
            var current = 0;
            foreach (var (obj, key) in fields)
            {
                if (obj?[key]?.GetValue<int>() is not int value || value <= cap) continue;
                current = Math.Max(current, value);
                obj[key] = cap;
            }
            if (current == 0) return;
            await File.WriteAllTextAsync(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }), ct);
            _logger.LogInformation("Capped '{Alias}' context {From} -> {To} to fit the KV cache in memory.", model.Alias, current, _options.MaxSequenceLength);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning(ex, "Could not cap max_length for '{Alias}'; loading with the model default.", model.Alias);
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _cts?.Cancel();
        try { await _loadTask.WaitAsync(cancellationToken); }
        catch (OperationCanceledException) { /* shutdown requested while a model was loading */ }
        _cts?.Dispose();
        Embeddings?.Dispose();
        if (!FoundryLocalManager.IsInitialized) return;

        var manager = FoundryLocalManager.Instance;
        try
        {
            var catalog = await manager.GetCatalogAsync(cancellationToken);
            foreach (var loaded in await catalog.GetLoadedModelsAsync(cancellationToken))
                await loaded.UnloadAsync(cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error while unloading models during shutdown.");
        }
        finally
        {
            manager.Shutdown();
            manager.Dispose();
        }
    }
}
