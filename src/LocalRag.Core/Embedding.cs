using Microsoft.Extensions.AI;

namespace LocalRag.Core;

internal static class EmbeddingHelpers
{
    /// <summary>
    /// L2-normalises in place. With unit vectors, cosine similarity equals the dot product, so scores are
    /// comparable across chunks and a single threshold (<see cref="RagOptions.MinSimilarity"/>) is meaningful.
    /// </summary>
    public static float[] Normalize(ReadOnlyMemory<float> vector)
    {
        var array = vector.ToArray();
        double sum = 0;
        foreach (var v in array) sum += (double)v * v;
        var norm = (float)Math.Sqrt(sum);
        if (norm > 0) for (var i = 0; i < array.Length; i++) array[i] /= norm;
        return array;
    }

    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length) throw new InvalidOperationException($"Embedding dimension mismatch ({a.Length} vs {b.Length}).");
        float sum = 0;
        for (var i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    public static string ModelId(IEmbeddingGenerator<string, Embedding<float>> generator, RagOptions options) =>
        generator.GetService<EmbeddingGeneratorMetadata>()?.DefaultModelId ?? options.EmbeddingModelAlias;
}
