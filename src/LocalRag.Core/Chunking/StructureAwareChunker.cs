using System.Text;
using System.Text.RegularExpressions;

namespace LocalRag.Core.Chunking;

/// <summary>
/// Splits on headings (Markdown "#", or short numbered lines like "3. Warranty"), then packs whole paragraphs
/// into chunks up to the size limit. Every chunk is prefixed with its heading path so the embedding and the
/// model both know which section it came from. Paragraphs longer than the limit fall back to a sliding window.
/// </summary>
public sealed partial class StructureAwareChunker(int sizeWords, int overlapWords) : IChunker
{
    [GeneratedRegex(@"^(#{1,6})\s+(.+?)\s*#*\s*$")]
    private static partial Regex MarkdownHeading();

    [GeneratedRegex(@"^(\d+(?:\.\d+)*)\.?\s+\p{Lu}[^.!?]{0,80}$")]
    private static partial Regex NumberedHeading();

    private sealed class Section(string path)
    {
        public string Path { get; } = path;
        public List<string> Paragraphs { get; } = [];
    }

    public IReadOnlyList<TextChunk> Chunk(string text)
    {
        var chunks = new List<TextChunk>();
        var sections = Split(text);
        PromoteTitle(sections);
        foreach (var section in sections)
            Pack(section, chunks);
        return chunks;
    }

    /// <summary>
    /// A short heading-less preamble ("Acme Handbook") would become a junk chunk that matches every query. Treat it as the
    /// document title instead and prefix it to each section path, which also gives every chunk document-level context.
    /// </summary>
    private static void PromoteTitle(List<Section> sections)
    {
        if (sections.Count < 2 || sections[0].Path.Length > 0) return;
        var title = string.Join(" ", sections[0].Paragraphs);
        if (title.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length > 15) return;

        sections.RemoveAt(0);
        for (var i = 0; i < sections.Count; i++)
        {
            var promoted = new Section($"{title} › {sections[i].Path}");
            promoted.Paragraphs.AddRange(sections[i].Paragraphs);
            sections[i] = promoted;
        }
    }

    private static List<Section> Split(string text)
    {
        var sections = new List<Section>();
        var headingStack = new List<(int Level, string Title)>();
        var current = new Section(string.Empty);
        var paragraph = new StringBuilder();

        void FlushParagraph()
        {
            if (paragraph.Length == 0) return;
            current.Paragraphs.Add(paragraph.ToString().Trim());
            paragraph.Clear();
        }

        void StartSection(int level, string title)
        {
            FlushParagraph();
            if (current.Paragraphs.Count > 0) sections.Add(current);
            headingStack.RemoveAll(h => h.Level >= level);
            headingStack.Add((level, title));
            current = new Section(string.Join(" › ", headingStack.Select(h => h.Title)));
        }

        var lines = text.Replace("\r\n", "\n").Split('\n');
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (line.Length == 0) { FlushParagraph(); continue; }

            var md = MarkdownHeading().Match(line);
            if (md.Success) { StartSection(md.Groups[1].Length, md.Groups[2].Value); continue; }

            // "3. Warranty" style headings in plain text/PDF: short, start with a capital, and body text follows.
            var numbered = NumberedHeading().Match(line);
            if (numbered.Success && paragraph.Length == 0 && i + 1 < lines.Length && lines[i + 1].Trim().Length > 0)
            {
                var level = numbered.Groups[1].Value.Count(c => c == '.') + 1;
                StartSection(Math.Min(level, 6), line.TrimEnd('.'));
                continue;
            }

            if (paragraph.Length > 0) paragraph.Append(' ');
            paragraph.Append(line);
        }

        FlushParagraph();
        if (current.Paragraphs.Count > 0) sections.Add(current);
        return sections;
    }

    private void Pack(Section section, List<TextChunk> output)
    {
        var heading = section.Path.Length > 0 ? section.Path : null;
        var prefix = heading is null ? string.Empty : heading + "\n";
        var prefixWords = heading is null ? 0 : heading.Split(' ').Length;
        var budget = Math.Max(20, sizeWords - prefixWords);

        var buffer = new List<string>();
        var bufferWords = 0;

        void Emit()
        {
            if (buffer.Count == 0) return;
            output.Add(new TextChunk(output.Count, prefix + string.Join("\n\n", buffer), heading));
            buffer.Clear();
            bufferWords = 0;
        }

        foreach (var paragraph in section.Paragraphs)
        {
            var words = paragraph.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (words.Length > budget)
            {
                Emit();
                foreach (var piece in FixedSizeChunker.Window(words, budget, overlapWords, heading))
                    output.Add(new TextChunk(output.Count, prefix + piece.Content, heading));
                continue;
            }

            if (bufferWords + words.Length > budget) Emit();
            buffer.Add(paragraph);
            bufferWords += words.Length;
        }

        Emit();
    }
}
