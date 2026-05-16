package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/c3d4r/semsearch/internal/embed"
	"github.com/c3d4r/semsearch/internal/model"
	"github.com/c3d4r/semsearch/internal/search"
	"github.com/c3d4r/semsearch/internal/store"
)

var (
	numResults  int
	noSnippet   bool
	searchIndexDir string
)

var searchCmd = &cobra.Command{
	Use:   "search <query>",
	Short: "Search the index semantically",
	Long:  `Embed the query, search the HNSW index for nearest neighbors, and display matching file snippets.`,
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		query := args[0]

		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("get cwd: %w", err)
		}

		idxDir := searchIndexDir
		if idxDir == "" {
			idxDir = filepath.Join(cwd, ".semsearch")
		}

		st, err := store.NewSQLiteStore(idxDir)
		if err != nil {
			return fmt.Errorf("open store (run 'semsearch index' first): %w", err)
		}
		defer st.Close()

		hsw, err := store.NewHNSWStore(idxDir, 384)
		if err != nil {
			return fmt.Errorf("open HNSW store: %w", err)
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

		results, err := search.Run(query, numResults, !noSnippet, emb, tok, st, hsw)
		if err != nil {
			return fmt.Errorf("search: %w", err)
		}

		search.FormatResults(results, query)
		return nil
	},
}

func init() {
	searchCmd.Flags().IntVarP(&numResults, "num", "n", 10, "Number of results to return")
	searchCmd.Flags().BoolVar(&noSnippet, "no-snippet", false, "Don't show snippet preview")
	searchCmd.Flags().StringVar(&searchIndexDir, "index-dir", "", "Path to the index directory")

	rootCmd.AddCommand(searchCmd)
}
