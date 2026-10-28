using System.Runtime.CompilerServices;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.AI;
using MeaiChatMessage = Microsoft.Extensions.AI.ChatMessage;

namespace LocalRag.Foundry;

/// <summary>Adapts a Foundry Local <see cref="ChatSession"/> to <see cref="IChatClient"/>, so the RAG code never sees Foundry types.</summary>
internal sealed class FoundryChatClient(FoundryLocalService foundry) : IChatClient
{
    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<MeaiChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
        await GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken);

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<MeaiChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foundry.EnsureReady(chat: true);

        // A session keeps its own turn history and sampling options, so each request gets a fresh one: concurrent
        // requests can never see each other's settings or messages.
        using var session = new ChatSession(foundry.ChatModel!);
        session.SetStreaming(true);
        session.SetOptions(new RequestOptions
        {
            Search = new SearchOptions { MaxOutputTokens = options?.MaxOutputTokens, Temperature = options?.Temperature },        });

        using var request = new Request();
        foreach (var message in messages) request.AddItem(ToFoundry(message));

        await using var stream = session.ProcessStreamingRequestAsync(request, cancellationToken);
        await foreach (var item in stream.WithCancellation(cancellationToken))
        {
            // Reasoning models (deepseek-r1, phi-4-reasoning) tag their chain of thought; only answer text is streamed on.
            if (item is TextItem { Type: TextItemType.Default } text && !string.IsNullOrEmpty(text.Text))
                yield return new ChatResponseUpdate(ChatRole.Assistant, text.Text);
        }
        using var final = await stream.FinalResponse;
    }

    private static MessageItem ToFoundry(MeaiChatMessage m)
    {
        var text = m.Text ?? string.Empty;
        if (m.Role == ChatRole.System) return MessageItem.System(text);
        if (m.Role == ChatRole.Assistant) return MessageItem.Assistant(text);
        return MessageItem.User(text);
    }

    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this
        : serviceKey is null && serviceType == typeof(ChatClientMetadata) ? new ChatClientMetadata("foundry-local", defaultModelId: foundry.Chat.Alias)
        : null;

    public void Dispose() { }
}
