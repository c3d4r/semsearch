package store

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSQLiteStore(t *testing.T) {
	dir := t.TempDir()
	st, err := NewSQLiteStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	f := FileRecord{
		FilePath:   "/test/file.txt",
		FileHash:   "abc123",
		FileSize:   100,
		ChunkCount: 2,
		IndexedAt:  "2024-01-01T00:00:00Z",
	}

	if err := st.UpsertFile(f); err != nil {
		t.Fatal(err)
	}

	got, err := st.GetFile("/test/file.txt")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Fatal("expected file record, got nil")
	}
	if got.FileHash != "abc123" {
		t.Errorf("hash = %q, want abc123", got.FileHash)
	}

	missing, err := st.GetFile("/nonexistent.txt")
	if err != nil {
		t.Fatal(err)
	}
	if missing != nil {
		t.Error("expected nil for missing file")
	}
}

func TestChunks(t *testing.T) {
	dir := t.TempDir()
	st, err := NewSQLiteStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	chunks := []Chunk{
		{FilePath: "/a.txt", FileHash: "h1", ChunkIdx: 0, StartByte: 0, EndByte: 10, Text: "hello world", Mtime: 1000},
		{FilePath: "/a.txt", FileHash: "h1", ChunkIdx: 1, StartByte: 11, EndByte: 20, Text: "foo bar", Mtime: 1000},
	}

	ids, err := st.InsertChunks(chunks)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 ids, got %d", len(ids))
	}

	result, err := st.GetChunksByIDs(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(result))
	}
	if result[ids[0]].Text != "hello world" {
		t.Errorf("chunk 0 text = %q", result[ids[0]].Text)
	}
}

func TestDeleteFileChunks(t *testing.T) {
	dir := t.TempDir()
	st, err := NewSQLiteStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	chunks := []Chunk{
		{FilePath: "/b.txt", FileHash: "h2", ChunkIdx: 0, StartByte: 0, EndByte: 5, Text: "test", Mtime: 2000},
	}
	ids, err := st.InsertChunks(chunks)
	if err != nil {
		t.Fatal(err)
	}

	removedIDs, err := st.DeleteFileChunks("/b.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(removedIDs) != 1 {
		t.Fatalf("expected 1 removed id, got %d", len(removedIDs))
	}

	result, err := st.GetChunksByIDs(ids)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 0 {
		t.Errorf("expected 0 chunks after delete, got %d", len(result))
	}
}

func TestStats(t *testing.T) {
	dir := t.TempDir()
	st, err := NewSQLiteStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	chunks := []Chunk{
		{FilePath: "/x.txt", FileHash: "h1", ChunkIdx: 0, StartByte: 0, EndByte: 5, Text: "a", Mtime: 3000},
	}
	st.InsertChunks(chunks)
	st.UpsertFile(FileRecord{FilePath: "/x.txt", FileHash: "h1", FileSize: 5, ChunkCount: 1, IndexedAt: "now"})

	stats, err := st.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.FileCount != 1 {
		t.Errorf("file count = %d, want 1", stats.FileCount)
	}
	if stats.ChunkCount != 1 {
		t.Errorf("chunk count = %d, want 1", stats.ChunkCount)
	}
}

func TestHNSWStore(t *testing.T) {
	dir := t.TempDir()
	hsw, err := NewHNSWStore(dir, 3)
	if err != nil {
		t.Fatal(err)
	}

	if err := hsw.Add(1, []float32{1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := hsw.Add(2, []float32{0, 1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := hsw.Add(3, []float32{0, 0, 1}); err != nil {
		t.Fatal(err)
	}

	if hsw.Len() != 3 {
		t.Errorf("expected 3 nodes, got %d", hsw.Len())
	}

	results := hsw.Search([]float32{1, 0, 0}, 2)
	if len(results) < 1 {
		t.Error("expected results")
	}

	if results[0].Key != 1 {
		t.Errorf("expected nearest to be node 1, got %d", results[0].Key)
	}

	if err := hsw.Save(); err != nil {
		t.Fatal(err)
	}

	hsw2, err := NewHNSWStore(dir, 3)
	if err != nil {
		t.Fatal(err)
	}
	if hsw2.Len() != 3 {
		t.Errorf("after reload: expected 3 nodes, got %d", hsw2.Len())
	}

	if !hsw.Delete(1) {
		t.Error("failed to delete node 1")
	}
	if hsw.Len() != 2 {
		t.Errorf("after delete: expected 2 nodes, got %d", hsw.Len())
	}
}

func TestHNSWStoreWrongDim(t *testing.T) {
	dir := t.TempDir()
	hsw, err := NewHNSWStore(dir, 3)
	if err != nil {
		t.Fatal(err)
	}

	err = hsw.Add(1, []float32{1, 0})
	if err == nil {
		t.Error("expected error for wrong dimension, got nil")
	}
}

func TestDBFileCreated(t *testing.T) {
	dir := t.TempDir()
	_, err := NewSQLiteStore(dir)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(filepath.Join(dir, "index.db")); os.IsNotExist(err) {
		t.Error("index.db was not created")
	}
}
