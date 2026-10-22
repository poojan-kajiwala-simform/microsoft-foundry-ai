using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Options;

namespace LocalRag.Core.Storage;

/// <summary>A chunk as persisted: text, provenance and its (L2-normalised) embedding.</summary>
public sealed record StoredChunk(string Id, string Source, string? Section, int ChunkIndex, string Content, float[] Embedding);

/// <summary>
/// Durable source of truth (single SQLite file). The in-memory vector index is rebuilt from here at startup,
/// so a restart never needs the embedding model. Also records which embedding model/dimensions built the index.
/// </summary>
public sealed class SqliteChunkRepository
{
    public const string MetaEmbeddingModel = "embedding_model";
    public const string MetaDimensions = "dimensions";

    private readonly string _connectionString;

    public SqliteChunkRepository(IOptions<RagOptions> options)
    {
        var path = Path.GetFullPath(options.Value.DatabasePath);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        _connectionString = new SqliteConnectionStringBuilder { DataSource = path, Mode = SqliteOpenMode.ReadWriteCreate }.ToString();
    }

    private async Task<SqliteConnection> OpenAsync(CancellationToken ct)
    {
        var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync(ct);
        return connection;
    }

    public async Task EnsureCreatedAsync(CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var command = connection.CreateCommand();
        command.CommandText = """
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS Meta (Key TEXT PRIMARY KEY, Value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS Documents (
                Source TEXT PRIMARY KEY,
                Hash TEXT NOT NULL,
                ConfigSignature TEXT NOT NULL,
                ChunkCount INTEGER NOT NULL,
                IndexedUtc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS Chunks (
                Id TEXT PRIMARY KEY,
                Source TEXT NOT NULL REFERENCES Documents(Source) ON DELETE CASCADE,
                Section TEXT NULL,
                ChunkIndex INTEGER NOT NULL,
                Content TEXT NOT NULL,
                Embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS IX_Chunks_Source ON Chunks(Source);
            """;
        await command.ExecuteNonQueryAsync(ct);
    }

    public async Task<string?> GetMetaAsync(string key, CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT Value FROM Meta WHERE Key = $key";
        command.Parameters.AddWithValue("$key", key);
        return await command.ExecuteScalarAsync(ct) as string;
    }

    public async Task SetMetaAsync(string key, string value, CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO Meta (Key, Value) VALUES ($key, $value) ON CONFLICT(Key) DO UPDATE SET Value = $value";
        command.Parameters.AddWithValue("$key", key);
        command.Parameters.AddWithValue("$value", value);
        await command.ExecuteNonQueryAsync(ct);
    }

    public async Task<List<IndexedDocument>> GetDocumentsAsync(CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT Source, Hash, ConfigSignature, ChunkCount, IndexedUtc FROM Documents ORDER BY Source";
        var result = new List<IndexedDocument>();
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            result.Add(new IndexedDocument(
                reader.GetString(0), reader.GetString(1), reader.GetString(2), reader.GetInt32(3),
                DateTime.Parse(reader.GetString(4), null, System.Globalization.DateTimeStyles.RoundtripKind)));
        }
        return result;
    }

    /// <summary>Atomically swaps all chunks of one document: a crash mid-way leaves the previous version intact.</summary>
    public async Task ReplaceDocumentAsync(IndexedDocument document, IReadOnlyList<StoredChunk> chunks, CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var tx = (SqliteTransaction)await connection.BeginTransactionAsync(ct);

        await ExecuteAsync(connection, tx, "DELETE FROM Chunks WHERE Source = $s", ct, ("$s", document.Source));
        await ExecuteAsync(connection, tx,
            """
            INSERT INTO Documents (Source, Hash, ConfigSignature, ChunkCount, IndexedUtc) VALUES ($s, $h, $c, $n, $t)
            ON CONFLICT(Source) DO UPDATE SET Hash = $h, ConfigSignature = $c, ChunkCount = $n, IndexedUtc = $t
            """,
            ct, ("$s", document.Source), ("$h", document.Hash), ("$c", document.ConfigSignature),
            ("$n", document.ChunkCount), ("$t", document.IndexedUtc.ToString("O")));

        await using var insert = connection.CreateCommand();
        insert.Transaction = tx;
        insert.CommandText = "INSERT INTO Chunks (Id, Source, Section, ChunkIndex, Content, Embedding) VALUES ($id, $s, $sec, $i, $c, $e)";
        var id = insert.Parameters.Add("$id", SqliteType.Text);
        var source = insert.Parameters.Add("$s", SqliteType.Text);
        var section = insert.Parameters.Add("$sec", SqliteType.Text);
        var index = insert.Parameters.Add("$i", SqliteType.Integer);
        var content = insert.Parameters.Add("$c", SqliteType.Text);
        var embedding = insert.Parameters.Add("$e", SqliteType.Blob);
        foreach (var chunk in chunks)
        {
            id.Value = chunk.Id;
            source.Value = chunk.Source;
            section.Value = (object?)chunk.Section ?? DBNull.Value;
            index.Value = chunk.ChunkIndex;
            content.Value = chunk.Content;
            embedding.Value = ToBytes(chunk.Embedding);
            await insert.ExecuteNonQueryAsync(ct);
        }

        await tx.CommitAsync(ct);
    }

    public async Task DeleteDocumentAsync(string source, CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var tx = (SqliteTransaction)await connection.BeginTransactionAsync(ct);
        await ExecuteAsync(connection, tx, "DELETE FROM Chunks WHERE Source = $s", ct, ("$s", source));
        await ExecuteAsync(connection, tx, "DELETE FROM Documents WHERE Source = $s", ct, ("$s", source));
        await tx.CommitAsync(ct);
    }

    public async Task ClearAsync(CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var tx = (SqliteTransaction)await connection.BeginTransactionAsync(ct);
        await ExecuteAsync(connection, tx, "DELETE FROM Chunks", ct);
        await ExecuteAsync(connection, tx, "DELETE FROM Documents", ct);
        await ExecuteAsync(connection, tx, "DELETE FROM Meta", ct);
        await tx.CommitAsync(ct);
    }

    public async IAsyncEnumerable<StoredChunk> LoadAllChunksAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
    {
        await using var connection = await OpenAsync(ct);
        await using var command = connection.CreateCommand();
        command.CommandText = "SELECT Id, Source, Section, ChunkIndex, Content, Embedding FROM Chunks";
        await using var reader = await command.ExecuteReaderAsync(ct);
        while (await reader.ReadAsync(ct))
        {
            yield return new StoredChunk(
                reader.GetString(0), reader.GetString(1), reader.IsDBNull(2) ? null : reader.GetString(2),
                reader.GetInt32(3), reader.GetString(4), ToFloats((byte[])reader["Embedding"]));
        }
    }

    private static async Task ExecuteAsync(SqliteConnection c, SqliteTransaction tx, string sql, CancellationToken ct, params (string Name, object Value)[] parameters)
    {
        await using var command = c.CreateCommand();
        command.Transaction = tx;
        command.CommandText = sql;
        foreach (var (name, value) in parameters) command.Parameters.AddWithValue(name, value);
        await command.ExecuteNonQueryAsync(ct);
    }

    private static byte[] ToBytes(float[] vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] ToFloats(byte[] bytes)
    {
        var vector = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, vector, 0, bytes.Length);
        return vector;
    }
}
