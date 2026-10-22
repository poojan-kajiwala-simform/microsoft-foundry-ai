using Microsoft.Extensions.VectorData;

namespace LocalRag.Core.Indexing;

/// <summary>
/// The record shape stored in the vector collection. The vector dimension is not known at compile time
/// (it depends on the configured embedding model), so the schema is declared at runtime in
/// <see cref="VectorIndex"/> via a <see cref="VectorStoreCollectionDefinition"/> instead of attributes.
/// </summary>
public sealed class ChunkRecord
{
    public string Id { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public string? Section { get; set; }
    public int ChunkIndex { get; set; }
    public string Content { get; set; } = string.Empty;
    public ReadOnlyMemory<float> Embedding { get; set; }
}
