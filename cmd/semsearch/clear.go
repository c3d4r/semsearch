package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

var clearIndexDir string

var clearCmd = &cobra.Command{
	Use:   "clear",
	Short: "Remove the search index",
	RunE: func(cmd *cobra.Command, args []string) error {
		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("get cwd: %w", err)
		}

		idxDir := clearIndexDir
		if idxDir == "" {
			idxDir = filepath.Join(cwd, ".semsearch")
		}

		if err := os.RemoveAll(idxDir); err != nil {
			return fmt.Errorf("remove index: %w", err)
		}

		fmt.Printf("Index removed: %s\n", idxDir)
		return nil
	},
}

func init() {
	clearCmd.Flags().StringVar(&clearIndexDir, "index-dir", "", "Path to the index directory")
	rootCmd.AddCommand(clearCmd)
}
