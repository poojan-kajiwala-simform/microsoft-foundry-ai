namespace LocalRag.Core;

/// <summary>A model is still downloading/loading, or failed to load. Map to HTTP 503.</summary>
public sealed class ModelNotReadyException(string model, string state, string? detail = null)
    : Exception($"Model '{model}' is not ready (state: {state}){(detail is null ? "" : $": {detail}")}.");

/// <summary>The index was built with a different embedding model than the one configured now. Map to HTTP 409.</summary>
public sealed class ReindexRequiredException(string indexedWith, string configured)
    : Exception($"The index was built with embedding model '{indexedWith}' but '{configured}' is configured. " +
                "Vectors from different models are not comparable; run ingestion again to rebuild the index.");

public sealed class IngestionBusyException() : Exception("An ingestion run is already in progress.");
