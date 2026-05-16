package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "semsearch",
	Short: "Index and semantically search a local filesystem",
	Long: `semsearch builds a semantic search index over a local directory tree.

It extracts text from files, chunks it, generates embeddings using an
ONNX model (all-MiniLM-L6-v2), and stores vectors in an HNSW index
for fast approximate nearest-neighbor search.`,
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
