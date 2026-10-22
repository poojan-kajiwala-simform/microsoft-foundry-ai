using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using LocalRag.Core.Retrieval;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

namespace LocalRag.Core.Generation;

/// <summary>Query pipeline: retrieve → confidence gate → grounded prompt → streamed generation → citation check.</summary>
public sealed partial class RagAnswerService(
    Retriever retriever,
    PromptBuilder promptBuilder,
    IChatClient chat,
    IOptions<RagOptions> options)
{
    private readonly RagOptions _options = options.Value;

    [GeneratedRegex(@"\[(\d+)\]")]
    private static partial Regex CitationMarker();

    public const string NoDocumentsMessage = "No documents have been indexed yet. Ingest some documents first.";

    public async IAsyncEnumerable<RagEvent> AskAsync(
        string question, string? sourcePrefix = null, [EnumeratorCancellation] CancellationToken ct = default)
    {
        var total = Stopwatch.StartNew();
        var retrieval = await retriever.RetrieveAsync(question, sourcePrefix, ct);
        var retrievalTime = total.Elapsed;

        if (retrieval.IndexEmpty)
        {
            yield return new DoneEvent(NoDocumentsMessage, [], [], AbstainReasons.NoDocuments, new(retrievalTime, null, total.Elapsed, 0));
            yield break;
        }

        // Weak retrieval is the main cause of small-model hallucination: decline instead of asking the model to improvise.
        if (!retrieval.Confident)
        {
            yield return new DoneEvent(PromptBuilder.AbstainPhrase, [], [], AbstainReasons.LowConfidence, new(retrievalTime, null, total.Elapsed, 0));
            yield break;
        }

        var prompt = promptBuilder.Build(question, retrieval.Chunks);
        var citations = prompt.Included
            .Select((c, i) => new Citation(i + 1, c.Chunk.Source, c.Chunk.Section, Math.Round(c.Similarity, 3), c.Chunk.Content))
            .ToList();

        var chatOptions = new ChatOptions { MaxOutputTokens = _options.MaxAnswerTokens, Temperature = _options.Temperature };
        var answer = new StringBuilder();
        TimeSpan? firstToken = null;
        var fragments = 0;
        var sourcesSent = false;

        await foreach (var update in chat.GetStreamingResponseAsync(prompt.Messages, chatOptions, ct))
        {
            if (string.IsNullOrEmpty(update.Text)) continue;

            // Sources are announced once generation has actually started, so a model that isn't loaded fails
            // before anything is streamed (the caller can still return a clean 503).
            if (!sourcesSent) { sourcesSent = true; yield return new RetrievedEvent(citations); }

            firstToken ??= total.Elapsed;
            fragments++;
            answer.Append(update.Text);
            yield return new TokenEvent(update.Text);
        }

        var text = answer.ToString();
        var cited = CitationMarker().Matches(text)
            .Select(m => int.Parse(m.Groups[1].Value))
            .Where(n => n >= 1 && n <= citations.Count)
            .Distinct().Order().ToList();
        var declined = text.Contains("don't have enough information", StringComparison.OrdinalIgnoreCase);

        yield return new DoneEvent(text, citations, cited, declined ? AbstainReasons.ModelDeclined : null,
            new(retrievalTime, firstToken, total.Elapsed, fragments));
    }
}
