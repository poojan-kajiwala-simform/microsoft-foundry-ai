namespace LocalRag.Core.Chunking;

/// <summary>Sliding window over words. Cheap and predictable, but cuts sentences and sections wherever the window ends.</summary>
public sealed class FixedSizeChunker(int sizeWords, int overlapWords) : IChunker
{
    public IReadOnlyList<TextChunk> Chunk(string text) =>
        Window(text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries), sizeWords, overlapWords, section: null);

    internal static List<TextChunk> Window(string[] words, int size, int overlap, string? section)
    {
        var chunks = new List<TextChunk>();
        if (words.Length == 0) return chunks;

        var step = Math.Max(1, size - overlap);
        for (var start = 0; start < words.Length; start += step)
        {
            var length = Math.Min(size, words.Length - start);
            chunks.Add(new TextChunk(chunks.Count, string.Join(' ', words, start, length), section));
            if (start + length >= words.Length) break;
        }
        return chunks;
    }
}
