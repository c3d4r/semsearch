package store

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const schema = `
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT NOT NULL,
    file_hash   TEXT NOT NULL,
    chunk_idx   INTEGER NOT NULL,
    start_byte  INTEGER NOT NULL,
    end_byte    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    mtime       REAL NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    file_path   TEXT PRIMARY KEY,
    file_hash   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(file_hash);
`

type Chunk struct {
	ID        int64
	FilePath  string
	FileHash  string
	ChunkIdx  int
	StartByte int64
	EndByte   int64
	Text      string
	Mtime     float64
	IndexedAt string
}

type FileRecord struct {
	FilePath   string
	FileHash   string
	FileSize   int64
	ChunkCount int
	IndexedAt  string
}

type SQLiteStore struct {
	db *sql.DB
}

func NewSQLiteStore(indexDir string) (*SQLiteStore, error) {
	if err := os.MkdirAll(indexDir, 0755); err != nil {
		return nil, fmt.Errorf("create index dir: %w", err)
	}

	dbPath := filepath.Join(indexDir, "index.db")
	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL")
	if err != nil {
		return nil, fmt.Errorf("open database: %w", err)
	}

	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("create schema: %w", err)
	}

	return &SQLiteStore{db: db}, nil
}

func (s *SQLiteStore) Close() error {
	return s.db.Close()
}

func (s *SQLiteStore) GetFile(filePath string) (*FileRecord, error) {
	row := s.db.QueryRow(
		"SELECT file_path, file_hash, file_size, chunk_count, indexed_at FROM files WHERE file_path = ?",
		filePath,
	)
	var f FileRecord
	err := row.Scan(&f.FilePath, &f.FileHash, &f.FileSize, &f.ChunkCount, &f.IndexedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func (s *SQLiteStore) UpsertFile(f FileRecord) error {
	_, err := s.db.Exec(
		`INSERT INTO files (file_path, file_hash, file_size, chunk_count, indexed_at)
		 VALUES (?, ?, ?, ?, ?)
		 ON CONFLICT(file_path) DO UPDATE SET
		   file_hash = excluded.file_hash,
		   file_size = excluded.file_size,
		   chunk_count = excluded.chunk_count,
		   indexed_at = excluded.indexed_at`,
		f.FilePath, f.FileHash, f.FileSize, f.ChunkCount, f.IndexedAt,
	)
	return err
}

func (s *SQLiteStore) DeleteFileChunks(filePath string) ([]int64, error) {
	rows, err := s.db.Query("SELECT id FROM chunks WHERE file_path = ?", filePath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}

	if _, err := s.db.Exec("DELETE FROM chunks WHERE file_path = ?", filePath); err != nil {
		return nil, err
	}
	return ids, rows.Err()
}

func (s *SQLiteStore) InsertChunks(chunks []Chunk) ([]int64, error) {
	tx, err := s.db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(
		`INSERT INTO chunks (file_path, file_hash, chunk_idx, start_byte, end_byte, text, mtime, indexed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
	)
	if err != nil {
		return nil, err
	}
	defer stmt.Close()

	now := time.Now().UTC().Format(time.RFC3339)
	var ids []int64

	for _, ch := range chunks {
		result, err := stmt.Exec(ch.FilePath, ch.FileHash, ch.ChunkIdx, ch.StartByte, ch.EndByte, ch.Text, ch.Mtime, now)
		if err != nil {
			return nil, fmt.Errorf("insert chunk: %w", err)
		}
		id, _ := result.LastInsertId()
		ids = append(ids, id)
	}

	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return ids, nil
}

func (s *SQLiteStore) GetChunksByIDs(ids []int64) (map[int64]Chunk, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	query := "SELECT id, file_path, file_hash, chunk_idx, start_byte, end_byte, text, mtime, indexed_at FROM chunks WHERE id IN ("
	args := make([]interface{}, len(ids))
	for i, id := range ids {
		if i > 0 {
			query += ","
		}
		query += "?"
		args[i] = id
	}
	query += ")"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	result := make(map[int64]Chunk)
	for rows.Next() {
		var ch Chunk
		if err := rows.Scan(&ch.ID, &ch.FilePath, &ch.FileHash, &ch.ChunkIdx, &ch.StartByte, &ch.EndByte, &ch.Text, &ch.Mtime, &ch.IndexedAt); err != nil {
			return nil, err
		}
		result[ch.ID] = ch
	}
	return result, rows.Err()
}

type Stats struct {
	FileCount  int
	ChunkCount int
	TotalBytes int64
}

func (s *SQLiteStore) Stats() (Stats, error) {
	var st Stats
	if err := s.db.QueryRow("SELECT COUNT(*) FROM files").Scan(&st.FileCount); err != nil {
		return st, err
	}
	if err := s.db.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&st.ChunkCount); err != nil {
		return st, err
	}
	s.db.QueryRow("SELECT COALESCE(SUM(file_size), 0) FROM files").Scan(&st.TotalBytes)
	return st, nil
}

func (s *SQLiteStore) DeleteAll() error {
	_, err := s.db.Exec("DELETE FROM chunks; DELETE FROM files;")
	return err
}
