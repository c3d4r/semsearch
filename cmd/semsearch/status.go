package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/c3d4r/semsearch/internal/store"
)

var (
	statusIndexDir string
)

var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Show index statistics",
	RunE: func(cmd *cobra.Command, args []string) error {
		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("get cwd: %w", err)
		}

		idxDir := statusIndexDir
		if idxDir == "" {
			idxDir = filepath.Join(cwd, ".semsearch")
		}

		st, err := store.NewSQLiteStore(idxDir)
		if err != nil {
			return fmt.Errorf("open store (run 'semsearch index' first): %w", err)
		}
		defer st.Close()

		stats, err := st.Stats()
		if err != nil {
			return fmt.Errorf("get stats: %w", err)
		}

		hsw, err := store.NewHNSWStore(idxDir, 384)
		if err != nil {
			fmt.Printf("Warning: could not open HNSW index: %v\n", err)
		}

		fmt.Printf("Index location: %s\n", idxDir)
		fmt.Printf("  Files indexed: %d\n", stats.FileCount)
		fmt.Printf("  Chunks indexed: %d\n", stats.ChunkCount)

		if hsw != nil {
			fmt.Printf("  Vectors in HNSW: %d\n", hsw.Len())
		}

		totalMB := float64(stats.TotalBytes) / 1024 / 1024
		fmt.Printf("  Total source size: %.1f MB\n", totalMB)

		dbPath := filepath.Join(idxDir, "index.db")
		hnswPath := filepath.Join(idxDir, "index.hnsw")
		var indexSize int64
		for _, p := range []string{dbPath, hnswPath} {
			if fi, err := os.Stat(p); err == nil {
				indexSize += fi.Size()
			}
		}
		fmt.Printf("  Index size on disk: %.1f MB\n", float64(indexSize)/1024/1024)

		return nil
	},
}

func init() {
	statusCmd.Flags().StringVar(&statusIndexDir, "index-dir", "", "Path to the index directory")
	rootCmd.AddCommand(statusCmd)
}
