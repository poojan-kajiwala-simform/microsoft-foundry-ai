using System.Runtime.CompilerServices;
using Betalgo.Ranul.OpenAI.ObjectModels.RequestModels;
using Microsoft.Extensions.AI;
using MeaiChatMessage = Microsoft.Extensions.AI.ChatMessage;

namespace LocalRag.Foundry;

/// <summary>Adapts Foundry Local's in-process chat client to <see cref="IChatClient"/>, so the RAG code never sees Foundry types.</summary>
internal sealed class FoundryChatClient(FoundryLocalService foundry) : IChatClient
{
    // The SDK client holds mutable per-client settings, so serialise calls that change them.
    private readonly SemaphoreSlim _settingsGate = new(1, 1);

    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<MeaiChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        await GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken);

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<MeaiChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foundry.EnsureReady(chat: true);
#pragma warning disable CS0618
        var client = foundry.ChatClient!;
#pragma warning restore CS0618

        var request = messages.Select(ToFoundry).ToList();

        await _settingsGate.WaitAsync(cancellationToken);
        try
        {
            if (options?.MaxOutputTokens is { } max) client.Settings.MaxTokens = max;
            if (options?.Temperature is { } temperature) client.Settings.Temperature = temperature;
        }
        finally { _settingsGate.Release(); }

        await foreach (var chunk in client.CompleteChatStreamingAsync(request, cancellationToken))
        {
            var text = chunk.Choices?.FirstOrDefault()?.Delta?.Content;
            if (!string.IsNullOrEmpty(text))
                yield return new ChatResponseUpdate(ChatRole.Assistant, text);
        }
    }

    private static Betalgo.Ranul.OpenAI.ObjectModels.RequestModels.ChatMessage ToFoundry(MeaiChatMessage m)
    {
        var text = m.Text ?? string.Empty;
        if (m.Role == ChatRole.System) return Betalgo.Ranul.OpenAI.ObjectModels.RequestModels.ChatMessage.FromSystem(text);
        if (m.Role == ChatRole.Assistant) return Betalgo.Ranul.OpenAI.ObjectModels.RequestModels.ChatMessage.FromAssistant(text);
        return Betalgo.Ranul.OpenAI.ObjectModels.RequestModels.ChatMessage.FromUser(text);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this
        : serviceKey is null && serviceType == typeof(ChatClientMetadata) ? new ChatClientMetadata("foundry-local", defaultModelId: foundry.Chat.Alias)
        : null;

    public void Dispose() => _settingsGate.Dispose();
}
