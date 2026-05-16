package main

import (
	"fmt"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/c3d4r/semsearch/internal/embed"
	"github.com/c3d4r/semsearch/internal/index"
	"github.com/c3d4r/semsearch/internal/model"
)

var (
	indexDir  string
	includes  []string
	excludes  []string
	forceReindex bool
)

var indexCmd = &cobra.Command{
	Use:   "index [directory]",
	Short: "Build a semantic search index over a directory tree",
	Long: `Walk the directory tree, extract text from files, chunk it,
generate embeddings, and store them for semantic search.

Supported file types: text, code, markdown, and PDF.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		root := "."
		if len(args) > 0 {
			root = args[0]
		}

		root, err := filepath.Abs(root)
		if err != nil {
			return fmt.Errorf("resolve path: %w", err)
		}

		idxDir := indexDir
		if idxDir == "" {
			idxDir = filepath.Join(root, ".semsearch")
		}

		if err := model.EnsureModels(); err != nil {
			return fmt.Errorf("ensure models: %w", err)
		}

		vocabPath, err := model.VocabPath()
		if err != nil {
			return err
		}

		tok, err := embed.NewTokenizer(vocabPath)
		if err != nil {
			return fmt.Errorf("load tokenizer: %w", err)
		}

		modelPath, err := model.ModelPath()
		if err != nil {
			return err
		}

		onnxLib, err := model.FindONNXLibrary()
		if err != nil {
			return err
		}

		emb, err := embed.NewONNXEmbedder(modelPath, onnxLib)
		if err != nil {
			return fmt.Errorf("load ONNX model: %w", err)
		}
		defer emb.Destroy()

		cfg := index.Config{
			Root:     root,
			IndexDir: idxDir,
			Includes: includes,
			Excludes: excludes,
			Force:    forceReindex,
		}

		return index.Run(cfg, emb, tok)
	},
}

func init() {
	indexCmd.Flags().StringVar(&indexDir, "index-dir", "", "Directory to store the index (default: .semsearch/ in root)")
	indexCmd.Flags().StringSliceVarP(&includes, "include", "i", nil, "File glob patterns to include (repeatable)")
	indexCmd.Flags().StringSliceVarP(&excludes, "exclude", "e", nil, "File glob patterns to exclude (repeatable)")
	indexCmd.Flags().BoolVarP(&forceReindex, "force", "f", false, "Force re-index all files")

	rootCmd.AddCommand(indexCmd)
}
