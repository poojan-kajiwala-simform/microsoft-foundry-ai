using LocalRag.Core;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace LocalRag.Tests;

public class IngestionTests
{
    private const string Refunds = "# Handbook\n\n## Refunds\n\nCustomers may request a full refund within thirty days of purchase.\n\n## Shipping\n\nStandard shipping takes five business days.";

    [Fact]
    public async Task Second_run_is_a_no_op_and_does_not_re_embed()
    {
        using var rig = new Rig();
        rig.Write("handbook.md", Refunds);

        var first = await rig.Ingestion.SyncFolderAsync(rig.Docs);
        var callsAfterFirst = rig.Embedder.Calls;
        var second = await rig.Ingestion.SyncFolderAsync(rig.Docs);

        Assert.Equal(1, first.Added);
        Assert.Equal(1, second.Unchanged);
        Assert.Equal(0, second.Chunks);
        Assert.Equal(callsAfterFirst, rig.Embedder.Calls);          // idempotent: nothing re-embedded
        Assert.Equal(first.Chunks, rig.Index.Count);                // and no duplicates in the index
    }

    [Fact]
    public async Task Changed_file_replaces_old_chunks_and_deleted_file_removes_them()
    {
        using var rig = new Rig();
        var path = rig.Write("handbook.md", Refunds);
        rig.Write("other.md", "# Other\n\nSomething about warranty claims and repairs.");
        await rig.Ingestion.SyncFolderAsync(rig.Docs);

        File.WriteAllText(path, "# Handbook\n\n## Refunds\n\nRefunds are now sixty days.");
        var updated = await rig.Ingestion.SyncFolderAsync(rig.Docs);
        Assert.Equal(1, updated.Updated);
        Assert.Equal(1, updated.Unchanged);

        var hits = await rig.Retriever.RetrieveAsync("refund days");
        Assert.Contains(hits.Chunks, c => c.Chunk.Content.Contains("sixty"));
        Assert.DoesNotContain(hits.Chunks, c => c.Chunk.Content.Contains("thirty"));

        File.Delete(path);
        var removed = await rig.Ingestion.SyncFolderAsync(rig.Docs);
        Assert.Equal(1, removed.Removed);
        Assert.Single((await rig.Repository.GetDocumentsAsync()));
        Assert.DoesNotContain((await rig.Retriever.RetrieveAsync("refund")).Chunks, c => c.Chunk.Source == "handbook.md");
    }

    [Fact]
    public async Task Same_filename_in_different_folders_does_not_collide()
    {
        using var rig = new Rig();
        rig.Write("a/policy.md", "# P\n\nalpha content about apples");
        rig.Write("b/policy.md", "# P\n\nbeta content about bananas");
        await rig.Ingestion.SyncFolderAsync(rig.Docs);

        var sources = (await rig.Repository.GetDocumentsAsync()).Select(d => d.Source).ToList();
        Assert.Equal(["a/policy.md", "b/policy.md"], sources);
    }

    [Fact]
    public async Task One_bad_file_is_reported_without_aborting_the_run()
    {
        using var rig = new Rig(o => o.MaxFileBytes = 2000);
        rig.Write("good.md", Refunds);
        File.WriteAllBytes(Path.Combine(rig.Docs, "broken.pdf"), [1, 2, 3, 4]);
        File.WriteAllText(Path.Combine(rig.Docs, "empty.txt"), "   ");
        File.WriteAllText(Path.Combine(rig.Docs, "huge.txt"), new string('x', 5000));

        var report = await rig.Ingestion.SyncFolderAsync(rig.Docs);

        Assert.Equal(1, report.Added);
        Assert.Equal(3, report.Failures.Count);
        Assert.Contains(report.Failures, f => f.Source == "broken.pdf" && f.Reason.Contains("parsed"));
        Assert.Contains(report.Failures, f => f.Source == "empty.txt" && f.Reason.Contains("No extractable"));
        Assert.Contains(report.Failures, f => f.Source == "huge.txt" && f.Reason.Contains("limit"));
    }

    [Fact]
    public async Task Too_many_chunks_is_rejected_and_keeps_the_previous_version()
    {
        using var rig = new Rig(o => { o.MaxChunksPerDocument = 3; o.ChunkSizeWords = 30; o.ChunkOverlapWords = 5; o.Chunking = ChunkingStrategy.Fixed; });
        var path = rig.Write("doc.txt", "alpha beta gamma");
        await rig.Ingestion.SyncFolderAsync(rig.Docs);

        File.WriteAllText(path, string.Join(' ', Enumerable.Range(0, 500).Select(i => $"word{i}")));
        var report = await rig.Ingestion.SyncFolderAsync(rig.Docs);

        Assert.Single(report.Failures);
        Assert.Equal(1, rig.Index.Count);                            // old single chunk still searchable
    }

    [Fact]
    public async Task Changing_the_embedding_model_forces_a_full_reindex()
    {
        using var rig = new Rig();
        rig.Write("handbook.md", Refunds);
        await rig.Ingestion.SyncFolderAsync(rig.Docs);

        using var rig2 = new Rig(embedder: new HashEmbedder("other-model"), dbPath: rig.Options.DatabasePath);
        rig2.Write("handbook.md", Refunds);
        await rig2.Repository.EnsureCreatedAsync();
        var report = await rig2.Ingestion.SyncFolderAsync(rig2.Docs);

        Assert.True(report.Reindexed);
        Assert.Equal(1, report.Added);                              // re-embedded even though the file hash is unchanged
    }

    [Fact]
    public async Task Querying_with_a_different_model_than_the_index_throws()
    {
        using var rig = new Rig();
        rig.Write("handbook.md", Refunds);
        await rig.Ingestion.SyncFolderAsync(rig.Docs);

        var mismatched = new LocalRag.Core.Retrieval.Retriever(new HashEmbedder("other-model"), rig.Index, rig.Repository,
            Microsoft.Extensions.Options.Options.Create(rig.Options));
        await Assert.ThrowsAsync<ReindexRequiredException>(() => mismatched.RetrieveAsync("refund"));
    }

    [Fact]
    public async Task Index_survives_a_restart_without_the_embedding_model()
    {
        using var rig = new Rig();
        rig.Write("handbook.md", Refunds);
        await rig.Ingestion.SyncFolderAsync(rig.Docs);

        // "Restart": fresh in-memory index hydrated from SQLite only.
        var fresh = new VectorIndex();
        var hydrator = new IndexHydrator(rig.Repository, fresh, NullLogger<IndexHydrator>.Instance);
        await hydrator.StartAsync(CancellationToken.None);

        Assert.Equal(rig.Index.Count, fresh.Count);
        var hits = await fresh.SearchAsync((await rig.Embedder.GenerateAsync(["refund"]))[0].Vector.ToArray(), 3);
        Assert.NotEmpty(hits);
    }
}

public class RetrievalAndAnswerTests
{
    private static async Task<Rig> Seeded(Action<RagOptions>? configure = null)
    {
        var rig = new Rig(configure);
        rig.Write("returns.md", "# Returns\n\n## Refunds\n\nCustomers may request a full refund within thirty days of purchase.");
        rig.Write("ops.md", "# Operations\n\n## Errors\n\nError code E4471 means the pump is jammed. Restart the pump controller.");
        rig.Write("hr.md", "# HR\n\n## Vacation\n\nEmployees accrue twenty vacation days per year.");
        await rig.Ingestion.SyncFolderAsync(rig.Docs);
        return rig;
    }

    [Fact]
    public async Task Relevant_question_is_confident_and_finds_the_right_section()
    {
        using var rig = await Seeded();
        var result = await rig.Retriever.RetrieveAsync("refund within thirty days");

        Assert.True(result.Confident);
        Assert.Equal("returns.md", result.Chunks[0].Chunk.Source);
        Assert.Equal("Returns › Refunds", result.Chunks[0].Chunk.Section);
    }

    [Fact]
    public async Task Unrelated_question_is_below_the_threshold()
    {
        using var rig = await Seeded();
        var result = await rig.Retriever.RetrieveAsync("quantum entanglement of neutron stars");
        Assert.False(result.Confident);
    }

    [Fact]
    public async Task Source_prefix_filters_candidates()
    {
        using var rig = await Seeded();
        var result = await rig.Retriever.RetrieveAsync("refund thirty days vacation pump", sourcePrefix: "hr.md");
        Assert.All(result.Chunks, c => Assert.Equal("hr.md", c.Chunk.Source));
    }

    [Fact]
    public async Task Hybrid_search_surfaces_an_exact_code_that_vectors_alone_rank_lower()
    {
        using var rig = await Seeded(o => { o.HybridSearch = true; o.TopK = 1; });
        var result = await rig.Retriever.RetrieveAsync("E4471");
        Assert.Equal("ops.md", result.Chunks[0].Chunk.Source);
        Assert.NotNull(result.Chunks[0].KeywordRank);
    }

    [Fact]
    public async Task Empty_index_reports_no_documents_without_calling_the_model()
    {
        using var rig = new Rig();
        var chat = new ScriptedChat("should not be called");
        var events = await rig.Answers(chat).AskAsync("anything?").ToListAsync();

        var done = Assert.IsType<DoneEvent>(Assert.Single(events));
        Assert.Equal(AbstainReasons.NoDocuments, done.AbstainReason);
        Assert.Equal(0, chat.Calls);
    }

    [Fact]
    public async Task Low_confidence_declines_without_calling_the_model()
    {
        using var rig = await Seeded();
        var chat = new ScriptedChat("should not be called");
        var events = await rig.Answers(chat).AskAsync("quantum entanglement of neutron stars").ToListAsync();

        var done = Assert.IsType<DoneEvent>(events[^1]);
        Assert.Equal(AbstainReasons.LowConfidence, done.AbstainReason);
        Assert.Equal(PromptBuilder.AbstainPhrase, done.Answer);
        Assert.Equal(0, chat.Calls);
    }

    [Fact]
    public async Task Grounded_answer_streams_reports_sources_and_extracts_citations()
    {
        using var rig = await Seeded();
        var chat = new ScriptedChat("A full refund ", "is available ", "within 30 days [1].");
        var events = await rig.Answers(chat).AskAsync("refund within thirty days").ToListAsync();

        var retrieved = Assert.IsType<RetrievedEvent>(events[0]);
        Assert.Equal("returns.md", retrieved.Citations[0].Source);
        Assert.Equal(3, events.OfType<TokenEvent>().Count());

        var done = Assert.IsType<DoneEvent>(events[^1]);
        Assert.Equal([1], done.CitedIndexes);
        Assert.Null(done.AbstainReason);
        Assert.NotNull(done.Timings.FirstToken);
        Assert.Contains("[1] (returns.md › Returns › Refunds)", chat.LastMessages![0].Text);
    }

    [Fact]
    public async Task Model_that_declines_is_flagged()
    {
        using var rig = await Seeded();
        var chat = new ScriptedChat(PromptBuilder.AbstainPhrase);
        var done = (DoneEvent)(await rig.Answers(chat).AskAsync("refund within thirty days").ToListAsync())[^1];
        Assert.Equal(AbstainReasons.ModelDeclined, done.AbstainReason);
    }

    [Fact]
    public async Task Citations_to_nonexistent_sources_are_ignored()
    {
        using var rig = await Seeded();
        var done = (DoneEvent)(await rig.Answers(new ScriptedChat("Made up [9].")).AskAsync("refund within thirty days").ToListAsync())[^1];
        Assert.Empty(done.CitedIndexes);
    }
}

internal static class AsyncExt
{
    public static async Task<List<T>> ToListAsync<T>(this IAsyncEnumerable<T> source)
    {
        var list = new List<T>();
        await foreach (var item in source) list.Add(item);
        return list;
    }
}
