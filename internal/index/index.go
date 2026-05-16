package index

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/cespare/xxhash/v2"
	"github.com/schollz/progressbar/v3"

	"github.com/c3d4r/semsearch/internal/embed"
	"github.com/c3d4r/semsearch/internal/store"
)

type Config struct {
	Root     string
	IndexDir string
	Includes []string
	Excludes []string
	Force    bool
}

func Run(cfg Config, emb *embed.ONNXEmbedder, tok *embed.Tokenizer) error {
	st, err := store.NewSQLiteStore(cfg.IndexDir)
	if err != nil {
		return fmt.Errorf("open store: %w", err)
	}
	defer st.Close()

	hsw, err := store.NewHNSWStore(cfg.IndexDir, embed.EmbeddingDim)
	if err != nil {
		return fmt.Errorf("open HNSW store: %w", err)
	}

	walker := NewWalker(cfg.Root)
	walker.SetIncludes(cfg.Includes)
	walker.SetExcludes(cfg.Excludes)

	entries, err := walker.Walk()
	if err != nil {
		return fmt.Errorf("walk directory: %w", err)
	}

	splitter := NewSplitter()
	skippedCount := 0

	bar := progressbar.Default(int64(len(entries)), "indexing")

	for _, entry := range entries {
		bar.Describe(entry.RelPath)

		if !cfg.Force {
			existing, err := st.GetFile(entry.Path)
			if err != nil {
				return fmt.Errorf("check file state: %w", err)
			}
			if existing != nil {
				curHash := hashFileBytes(entry.Path)
				if curHash == existing.FileHash {
					skippedCount++
					bar.Add(1)
					continue
				}
				removedIDs, err := st.DeleteFileChunks(entry.Path)
				if err != nil {
					return fmt.Errorf("delete old chunks: %w", err)
				}
				hsw.DeleteMany(removedIDs)
			}
		}

		text, err := ExtractText(entry.Path)
		if err != nil {
			bar.Add(1)
			continue
		}

		chunks := splitter.Split(text)
		if len(chunks) == 0 {
			bar.Add(1)
			continue
		}

		fileHash := hashFileBytes(entry.Path)
		mtime := float64(entry.Info.ModTime().Unix())

		storeChunks := make([]store.Chunk, len(chunks))
		embeddings := make([][]float32, len(chunks))

		for i, ch := range chunks {
			tokInput := tok.Encode(ch.Text)
			vec, err := emb.Embed(tokInput.InputIDs, tokInput.AttentionMask, tokInput.TokenTypeIDs)
			if err != nil {
				return fmt.Errorf("embed chunk: %w", err)
			}
			embeddings[i] = vec

			storeChunks[i] = store.Chunk{
				FilePath:  entry.Path,
				FileHash:  fileHash,
				ChunkIdx:  ch.Idx,
				StartByte: ch.StartByte,
				EndByte:   ch.EndByte,
				Text:      ch.Text,
				Mtime:     mtime,
			}
		}

		ids, err := st.InsertChunks(storeChunks)
		if err != nil {
			return fmt.Errorf("insert chunks: %w", err)
		}

		for i, id := range ids {
			if err := hsw.Add(int(id), embeddings[i]); err != nil {
				return fmt.Errorf("add to HNSW: %w", err)
			}
		}

		if err := st.UpsertFile(store.FileRecord{
			FilePath:   entry.Path,
			FileHash:   fileHash,
			FileSize:   entry.Info.Size(),
			ChunkCount: len(chunks),
			IndexedAt:  time.Now().UTC().Format(time.RFC3339),
		}); err != nil {
			return fmt.Errorf("upsert file: %w", err)
		}

		bar.Add(1)
	}

	bar.Finish()

	if err := hsw.Save(); err != nil {
		return fmt.Errorf("save HNSW index: %w", err)
	}

	stats, _ := st.Stats()
	fmt.Printf("\nIndexed %d files (%d skipped), %d chunks total\n",
		stats.FileCount, skippedCount, stats.ChunkCount)
	fmt.Printf("Index location: %s\n", filepath.Join(cfg.IndexDir, "index.db"))

	return nil
}

func hashFileBytes(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	h := xxhash.Sum64(data)
	return fmt.Sprintf("%016x", h)
}
