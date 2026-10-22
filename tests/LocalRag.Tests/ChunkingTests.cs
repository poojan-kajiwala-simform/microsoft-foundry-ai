using LocalRag.Core.Chunking;
using LocalRag.Core.Generation;
using LocalRag.Core.Indexing;
using LocalRag.Core.Retrieval;
using Xunit;

namespace LocalRag.Tests;

public class ChunkingTests
{
    private static string Words(int n, string prefix = "w") => string.Join(' ', Enumerable.Range(0, n).Select(i => $"{prefix}{i}"));

    [Fact]
    public void Fixed_windows_overlap_and_cover_everything()
    {
        var chunks = new FixedSizeChunker(100, 20).Chunk(Words(250));

        Assert.Equal(3, chunks.Count);
        Assert.StartsWith("w0 ", chunks[0].Content);
        Assert.StartsWith("w80 ", chunks[1].Content);            // 100 - 20 overlap
        Assert.EndsWith("w249", chunks[^1].Content);
        Assert.All(chunks, c => Assert.Null(c.Section));
    }

    [Fact]
    public void Fixed_empty_text_yields_nothing() => Assert.Empty(new FixedSizeChunker(100, 10).Chunk("   \n "));

    [Fact]
    public void Structured_splits_on_markdown_headings_and_tracks_the_path()
    {
        var text = "# Policy\n\nIntro text.\n\n## Refunds\n\nRefunds within 30 days.\n\n### Exceptions\n\nGift cards excluded.\n\n## Shipping\n\nTakes 5 days.";
        var chunks = new StructureAwareChunker(100, 10).Chunk(text);

        Assert.Contains(chunks, c => c.Section == "Policy › Refunds › Exceptions" && c.Content.Contains("Gift cards"));
        Assert.Contains(chunks, c => c.Section == "Policy › Shipping" && c.Content.Contains("5 days"));
        // The heading is part of the embedded text so the vector "knows" its section.
        Assert.All(chunks, c => Assert.StartsWith(c.Section!, c.Content));
    }

    [Fact]
    public void Structured_recognises_numbered_plain_text_headings()
    {
        var text = "1. Refund Policy\nCustomers may request a refund.\n\n2. Shipping Policy\nStandard shipping takes a week.";
        var chunks = new StructureAwareChunker(100, 10).Chunk(text);

        Assert.Equal(2, chunks.Count);
        Assert.Equal("1. Refund Policy", chunks[0].Section);
        Assert.Equal("2. Shipping Policy", chunks[1].Section);
    }

    [Fact]
    public void Structured_turns_a_short_preamble_into_the_document_title_instead_of_a_junk_chunk()
    {
        var text = "Acme Handbook\n\n1. Refund Policy\nRefunds within thirty days.\n\n2. Shipping Policy\nFive business days.";
        var chunks = new StructureAwareChunker(100, 10).Chunk(text);

        Assert.Equal(2, chunks.Count);
        Assert.Equal("Acme Handbook › 1. Refund Policy", chunks[0].Section);
    }

    [Fact]
    public void Structured_never_mixes_sections_but_splits_oversized_sections()
    {
        var text = $"# A\n\n{Words(60, "a")}\n\n{Words(60, "b")}\n\n# B\n\n{Words(10, "c")}";
        var chunks = new StructureAwareChunker(80, 10).Chunk(text);

        Assert.True(chunks.Count(c => c.Section == "A") >= 2);   // two 60-word paragraphs can't share an 80-word chunk
        Assert.DoesNotContain(chunks, c => c.Section == "B" && c.Content.Contains("a0"));
    }

    [Fact]
    public void Structured_falls_back_to_a_window_for_a_giant_paragraph()
    {
        var chunks = new StructureAwareChunker(100, 10).Chunk("# Big\n\n" + Words(450));
        Assert.True(chunks.Count >= 4);
        Assert.All(chunks, c => Assert.True(c.Content.Split(' ').Length <= 105));
    }
}

public class Bm25Tests
{
    [Fact]
    public void Exact_rare_term_outranks_generic_text()
    {
        var bm25 = new Bm25Index();
        bm25.Rebuild([("a", "general travel expense policy for employees"), ("b", "error code E-4471 means the pump is jammed"), ("c", "employees must submit expenses monthly")]);

        var hits = bm25.Search("what does E-4471 mean", 3);
        Assert.Equal("b", hits[0].Id);
    }

    [Fact]
    public void No_overlap_means_no_hits() => Assert.Empty(new Bm25Index().Also(i => i.Rebuild([("a", "alpha beta")])).Search("zzz", 3));
}

internal static class Ext { public static T Also<T>(this T t, Action<T> a) { a(t); return t; } }

public class PromptBuilderTests
{
    private static RetrievedChunk Chunk(int i, int words) => new(
        new ChunkRecord { Id = $"c{i}", Source = $"f{i}.md", Section = $"S{i}", Content = string.Join(' ', Enumerable.Repeat("word", words)) }, 0.8, 0.1, i, null);

    [Fact]
    public void Token_budget_drops_lower_ranked_chunks()
    {
        var builder = new PromptBuilder(Microsoft.Extensions.Options.Options.Create(new LocalRag.Core.RagOptions { MaxContextTokens = 500 }));
        var prompt = builder.Build("q?", [Chunk(1, 200), Chunk(2, 200), Chunk(3, 200)]); // ~250 tokens each

        Assert.Single(prompt.Included);
        Assert.Equal(2, prompt.DroppedForBudget);
    }

    [Fact]
    public void Always_keeps_the_best_chunk_even_if_it_alone_exceeds_the_budget()
    {
        var builder = new PromptBuilder(Microsoft.Extensions.Options.Options.Create(new LocalRag.Core.RagOptions { MaxContextTokens = 50 }));
        var prompt = builder.Build("q?", [Chunk(1, 500)]);
        Assert.Single(prompt.Included);
        Assert.True(prompt.Messages[0].Text!.Length < 1000); // truncated to fit
    }

    [Fact]
    public void Prompt_numbers_sources_and_includes_the_abstain_instruction()
    {
        var builder = new PromptBuilder(Microsoft.Extensions.Options.Options.Create(new LocalRag.Core.RagOptions()));
        var system = builder.Build("q?", [Chunk(1, 5), Chunk(2, 5)]).Messages[0].Text!;
        Assert.Contains("[1] (f1.md › S1)", system);
        Assert.Contains("[2] (f2.md › S2)", system);
        Assert.Contains(PromptBuilder.AbstainPhrase, system);
    }
}
