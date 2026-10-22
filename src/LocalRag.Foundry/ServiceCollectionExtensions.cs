using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace LocalRag.Foundry;

public static class FoundryServiceCollectionExtensions
{
    /// <summary>
    /// Registers Foundry Local as the <see cref="IChatClient"/> and embedding generator, plus the hosted service that
    /// manages the runtime and model lifecycle. Replace this one call to move to Azure OpenAI / Ollama / any other provider.
    /// </summary>
    public static IServiceCollection AddFoundryLocal(this IServiceCollection services)
    {
        services.AddSingleton<FoundryLocalService>();
        services.AddHostedService(sp => sp.GetRequiredService<FoundryLocalService>());
        services.AddSingleton<IChatClient>(sp => new FoundryChatClient(sp.GetRequiredService<FoundryLocalService>()));
        services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(sp => new FoundryEmbeddingGenerator(sp.GetRequiredService<FoundryLocalService>()));
        return services;
    }
}
