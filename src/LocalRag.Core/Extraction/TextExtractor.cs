using System.Text;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;

namespace LocalRag.Core.Extraction;

/// <summary>
/// Extracts text from .pdf, .docx, .txt and .md. DOCX heading styles are emitted as Markdown headings so the
/// structure-aware chunker can use them; PDFs only yield flat text (heading detection falls back to heuristics).
/// </summary>
public sealed class TextExtractor
{
    private static readonly string[] SupportedExtensions = [".pdf", ".docx", ".txt", ".md"];

    public static bool IsSupported(string filePath) =>
        SupportedExtensions.Contains(Path.GetExtension(filePath), StringComparer.OrdinalIgnoreCase);

    public Task<string> ExtractTextAsync(string filePath, CancellationToken ct = default)
    {
        return Path.GetExtension(filePath).ToLowerInvariant() switch
        {
            ".pdf" => Task.Run(() => ExtractPdf(filePath), ct),
            ".docx" => Task.Run(() => ExtractDocx(filePath), ct),
            _ => File.ReadAllTextAsync(filePath, ct),
        };
    }

    private static string ExtractPdf(string filePath)
    {
        using var pdf = new PdfDocument(new PdfReader(filePath));
        var sb = new StringBuilder();
        for (var i = 1; i <= pdf.GetNumberOfPages(); i++)
            sb.AppendLine(PdfTextExtractor.GetTextFromPage(pdf.GetPage(i)));
        return sb.ToString();
    }

    private static string ExtractDocx(string filePath)
    {
        using var doc = WordprocessingDocument.Open(filePath, isEditable: false);
        var body = doc.MainDocumentPart?.Document?.Body;
        if (body is null) return string.Empty;

        var sb = new StringBuilder();
        foreach (var paragraph in body.Descendants<Paragraph>())
        {
            var text = paragraph.InnerText;
            if (string.IsNullOrWhiteSpace(text)) { sb.AppendLine(); continue; }

            var style = paragraph.ParagraphProperties?.ParagraphStyleId?.Val?.Value;
            if (style is not null && style.StartsWith("Heading", StringComparison.OrdinalIgnoreCase)
                && int.TryParse(style["Heading".Length..], out var level))
            {
                sb.Append('#', Math.Clamp(level, 1, 6)).Append(' ');
            }
            sb.AppendLine(text);
            sb.AppendLine();
        }
        return sb.ToString();
    }
}
