namespace LocalRag.Core.Chunking;

/// <param name="Content">Text that gets embedded and shown to the model (includes the section heading for context).</param>
/// <param name="Section">Heading path such as "Refunds › Exceptions"; null when the chunker has no structure information.</param>
public sealed record TextChunk(int Index, string Content, string? Section);

public interface IChunker
{
    IReadOnlyList<TextChunk> Chunk(string text);
}
