package search

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/coder/hnsw"

	"github.com/c3d4r/semsearch/internal/embed"
	"github.com/c3d4r/semsearch/internal/store"
)

type Result struct {
	Score     float32
	FilePath  string
	Snippet   string
	StartByte int64
	EndByte   int64
}

func Run(query string, numResults int, showSnippet bool, emb *embed.ONNXEmbedder, tok *embed.Tokenizer, st *store.SQLiteStore, hsw *store.HNSWStore) ([]Result, error) {
	tokInput := tok.Encode(query)
	queryVec, err := emb.Embed(tokInput.InputIDs, tokInput.AttentionMask, tokInput.TokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	nodes := hsw.Search(queryVec, numResults*3)

	ids := make([]int64, len(nodes))
	nodeMap := make(map[int64]hnsw.Node[int])
	for _, n := range nodes {
		id := int64(n.Key)
		ids = append(ids, id)
		nodeMap[id] = n
	}

	chunkMap, err := st.GetChunksByIDs(ids)
	if err != nil {
		return nil, fmt.Errorf("lookup chunks: %w", err)
	}

	seen := make(map[string]bool)
	var results []Result

	for _, n := range nodes {
		if len(results) >= numResults {
			break
		}

		ch, ok := chunkMap[int64(n.Key)]
		if !ok {
			continue
		}

		if seen[ch.FilePath] {
			continue
		}
		seen[ch.FilePath] = true

		score := cosineSimilarity(queryVec, n.Value)

		r := Result{
			Score:     score,
			FilePath:  ch.FilePath,
			StartByte: ch.StartByte,
			EndByte:   ch.EndByte,
		}

		if showSnippet {
			r.Snippet = truncateSnippet(ch.Text, 200)
		}

		results = append(results, r)
	}

	return results, nil
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA < 1e-12 || normB < 1e-12 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

func truncateSnippet(text string, maxLen int) string {
	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, "\n", " ")
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen] + "..."
}

func FormatResults(results []Result, query string) {
	if len(results) == 0 {
		fmt.Println("No results found.")
		return
	}

	cwd, _ := os.Getwd()

	for i, r := range results {
		displayPath := r.FilePath
		if rel, err := filepath.Rel(cwd, r.FilePath); err == nil && !strings.HasPrefix(rel, "..") {
			displayPath = rel
		}

		fmt.Printf("\n%s %s (%s%.3f%s)\n",
			bold(fmt.Sprintf("%d.", i+1)), displayPath,
			dim(""), r.Score, dim(""))

		fmt.Print(dim(" ─────────────────────────────────────────────────────\n"))

		if r.Snippet != "" {
			queryLower := strings.ToLower(query)
			snippet := r.Snippet
			highlighted := highlightTerms(snippet, queryLower)
			fmt.Printf(" %s\n", highlighted)
		}
	}
}

func bold(s string) string {
	return "\033[1m" + s + "\033[0m"
}

func dim(s string) string {
	return "\033[2m" + s + "\033[0m"
}

func highlightTerms(text, query string) string {
	var result strings.Builder
	remain := text

	for {
		idx := strings.Index(strings.ToLower(remain), query)
		if idx < 0 {
			result.WriteString(remain)
			break
		}
		result.WriteString(remain[:idx])
		result.WriteString("\033[33m")
		result.WriteString(remain[idx : idx+len(query)])
		result.WriteString("\033[0m")
		remain = remain[idx+len(query):]
	}

	return result.String()
}
