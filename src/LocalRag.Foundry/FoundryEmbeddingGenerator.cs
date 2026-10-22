using Microsoft.Extensions.AI;

namespace LocalRag.Foundry;

/// <summary>Adapts Foundry Local's embedding client to <see cref="IEmbeddingGenerator{TInput,TEmbedding}"/>.</summary>
internal sealed class FoundryEmbeddingGenerator(FoundryLocalService foundry) : IEmbeddingGenerator<string, Embedding<float>>
{
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        foundry.EnsureReady(chat: false);
#pragma warning disable CS0618
        var client = foundry.EmbeddingClient!;
#pragma warning restore CS0618

        var inputs = values.ToList();
        var result = new GeneratedEmbeddings<Embedding<float>>(inputs.Count);
        if (inputs.Count == 0) return result;

        var response = await client.GenerateEmbeddingsAsync(inputs, cancellationToken);
        foreach (var item in response.Data.OrderBy(d => d.Index))
            result.Add(new Embedding<float>(item.Embedding.Select(d => (float)d).ToArray()));
        return result;
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this
        : serviceKey is null && serviceType == typeof(EmbeddingGeneratorMetadata) ? new EmbeddingGeneratorMetadata("foundry-local", defaultModelId: foundry.Embedding.Alias)
        : null;

    public void Dispose() { }
}
