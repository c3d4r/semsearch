package embed

import "math"

func MeanPool(lastHiddenState [][][]float32, attentionMask []int64) []float32 {
	if len(lastHiddenState) == 0 || len(lastHiddenState[0]) == 0 {
		return nil
	}

	seqLen := len(lastHiddenState[0])
	dim := len(lastHiddenState[0][0])

	embedding := make([]float32, dim)

	var maskSum float32
	for i := 0; i < seqLen; i++ {
		maskVal := float32(attentionMask[i])
		maskSum += maskVal
		for j := 0; j < dim; j++ {
			embedding[j] += lastHiddenState[0][i][j] * maskVal
		}
	}

	if maskSum == 0 {
		maskSum = 1
	}
	for j := 0; j < dim; j++ {
		embedding[j] /= maskSum
	}

	l2Normalize(embedding)
	return embedding
}

func l2Normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 1e-12 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}
