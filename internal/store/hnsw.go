package store

import (
	"fmt"
	"path/filepath"

	"github.com/coder/hnsw"
)

type HNSWStore struct {
	graph    *hnsw.SavedGraph[int]
	path     string
	dim      int
}

func NewHNSWStore(indexDir string, dim int) (*HNSWStore, error) {
	path := filepath.Join(indexDir, "index.hnsw")
	g, err := hnsw.LoadSavedGraph[int](path)
	if err != nil {
		return nil, fmt.Errorf("load HNSW graph: %w", err)
	}

	g.Distance = hnsw.CosineDistance
	g.M = 16
	g.EfSearch = 50

	return &HNSWStore{
		graph: g,
		path:  path,
		dim:   dim,
	}, nil
}

func (h *HNSWStore) Add(id int, vec []float32) error {
	if len(vec) != h.dim {
		return fmt.Errorf("expected %d dimensions, got %d", h.dim, len(vec))
	}
	h.graph.Add(hnsw.MakeNode(id, vec))
	return nil
}

func (h *HNSWStore) AddBatch(chunks []struct {
	ID  int
	Vec []float32
}) error {
	nodes := make([]hnsw.Node[int], 0, len(chunks))
	for _, c := range chunks {
		if len(c.Vec) != h.dim {
			return fmt.Errorf("chunk %d: expected %d dimensions, got %d", c.ID, h.dim, len(c.Vec))
		}
		nodes = append(nodes, hnsw.MakeNode(c.ID, c.Vec))
	}
	h.graph.Add(nodes...)
	return nil
}

func (h *HNSWStore) Delete(id int) bool {
	return h.graph.Delete(id)
}

func (h *HNSWStore) DeleteMany(ids []int64) {
	for _, id := range ids {
		h.graph.Delete(int(id))
	}
}

func (h *HNSWStore) Search(queryVec []float32, k int) []hnsw.Node[int] {
	return h.graph.Search(queryVec, k)
}

func (h *HNSWStore) Len() int {
	return h.graph.Len()
}

func (h *HNSWStore) Save() error {
	return h.graph.Save()
}
