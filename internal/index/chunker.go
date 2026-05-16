package index

import (
	"strings"
	"unicode/utf8"
)

const (
	defaultChunkSize    = 1500
	defaultChunkOverlap = 375
)

type Splitter struct {
	ChunkSize    int
	ChunkOverlap int
}

func NewSplitter() *Splitter {
	return &Splitter{
		ChunkSize:    defaultChunkSize,
		ChunkOverlap: defaultChunkOverlap,
	}
}

type Chunk struct {
	Text      string
	StartByte int64
	EndByte   int64
	Idx       int
}

func (s *Splitter) Split(text string) []Chunk {
	if len(text) == 0 {
		return nil
	}

	if utf8.RuneCountInString(text) <= s.ChunkSize {
		return []Chunk{{
			Text:      strings.TrimSpace(text),
			StartByte: 0,
			EndByte:   int64(len(text)),
			Idx:       0,
		}}
	}

	sentences := splitSentences(text)

	var chunks []Chunk
	var current []string
	currentLen := 0
	var byteOffset int64
	currentStart := int64(0)
	chunkIdx := 0

	for _, sent := range sentences {
		sentLen := utf8.RuneCountInString(sent)
		rawLen := int64(len(sent))

		if currentLen+sentLen > s.ChunkSize && currentLen > 0 {
			chunkText := strings.TrimSpace(strings.Join(current, " "))
			chunks = append(chunks, Chunk{
				Text:      chunkText,
				StartByte: currentStart,
				EndByte:   byteOffset,
				Idx:       chunkIdx,
			})
			chunkIdx++

			if s.ChunkOverlap > 0 && len(current) > 1 {
				overlapIdx := len(current) - 1
				overlapLen := 0
				for i := overlapIdx; i >= 0; i-- {
					rLen := utf8.RuneCountInString(current[i])
					if overlapLen+rLen > s.ChunkOverlap {
						break
					}
					overlapLen += rLen
					overlapIdx = i
				}
				current = current[overlapIdx:]
				currentLen = overlapLen

				for i := range current {
					if i == 0 {
						currentStart = byteOffset
						for j := overlapIdx; j < len(sentences) && j < overlapIdx+1; j++ {
							currentStart -= int64(len(sentences[j]))
						}
					}
				}
			} else {
				current = nil
				currentLen = 0
				currentStart = byteOffset
			}
		}

		current = append(current, sent)
		currentLen += sentLen
		byteOffset += rawLen
	}

	if len(current) > 0 {
		chunkText := strings.TrimSpace(strings.Join(current, " "))
		chunks = append(chunks, Chunk{
			Text:      chunkText,
			StartByte: currentStart,
			EndByte:   byteOffset,
			Idx:       chunkIdx,
		})
	}

	return chunks
}

func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)

	for i := 0; i < len(runes); i++ {
		current.WriteRune(runes[i])

		if isSentenceEnd(runes, i) {
			sent := strings.TrimSpace(current.String())
			if len(sent) > 0 {
				sentences = append(sentences, sent)
			}
			current.Reset()
		}
	}

	if current.Len() > 0 {
		sent := strings.TrimSpace(current.String())
		if len(sent) > 0 {
			sentences = append(sentences, sent)
		}
	}

	if len(sentences) == 0 {
		return []string{text}
	}

	return sentences
}

func isSentenceEnd(runes []rune, i int) bool {
	r := runes[i]
	if r != '.' && r != '!' && r != '?' {
		return false
	}
	if i+1 < len(runes) && runes[i+1] == '"' {
		i++
	}
	if i+2 < len(runes) && runes[i+1] == '"' && runes[i+2] == '"' {
		return false
	}
	if i+1 < len(runes) {
		next := runes[i+1]
		if next == '"' {
			if i+2 < len(runes) && (runes[i+2] == ' ' || runes[i+2] == '\n' || runes[i+2] == '\r' || runes[i+2] == '\t') {
				return true
			}
		}
		if next == ' ' || next == '\n' || next == '\r' || next == '\t' {
			if i+2 < len(runes) {
				afterSpace := runes[i+2]
				if afterSpace >= 'A' && afterSpace <= 'Z' {
					return true
				}
			}
			if i+1 < len(runes) && (next == '\n' || next == '\r') {
				return true
			}
		}
	}
	if i+1 >= len(runes) {
		return true
	}
	return r == '\n' && i+1 < len(runes) && runes[i+1] == '\n'
}
