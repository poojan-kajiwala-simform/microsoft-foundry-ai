using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.AI;

namespace LocalRag.Foundry;

/// <summary>Adapts Foundry Local's <see cref="EmbeddingsSession"/> to <see cref="IEmbeddingGenerator{TInput,TEmbedding}"/>.</summary>
internal sealed class FoundryEmbeddingGenerator(FoundryLocalService foundry) : IEmbeddingGenerator<string, Embedding<float>>
{
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        foundry.EnsureReady(chat: false);

        var inputs = values.ToList();
        var result = new GeneratedEmbeddings<Embedding<float>>(inputs.Count);
        if (inputs.Count == 0) return result;

        using var request = new Request();
        foreach (var input in inputs) request.AddItem(new TextItem(input)); // the request owns each item and disposes it

        using var response = await foundry.Embeddings!.ProcessRequestAsync(request, cancellationToken);
        // One tensor per input, in input order.
        foreach (var item in response)
            if (item is TensorItem tensor) result.Add(new Embedding<float>(tensor.ToArray<float>()));

        if (result.Count != inputs.Count)
            throw new InvalidOperationException($"Embedding model returned {result.Count} vectors for {inputs.Count} inputs.");
        return result;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this
        : serviceKey is null && serviceType == typeof(EmbeddingGeneratorMetadata) ? new EmbeddingGeneratorMetadata("foundry-local", defaultModelId: foundry.Embedding.Alias)
        : null;

    public void Dispose() { }
}
