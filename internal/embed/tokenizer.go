package embed

import (
	"bufio"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

const (
	tokenCLS    = "[CLS]"
	tokenSEP    = "[SEP]"
	tokenPAD    = "[PAD]"
	tokenUNK    = "[UNK]"
	maxSeqLen   = 512
	clsTokenID  = 101
	sepTokenID  = 102
	padTokenID  = 0
	unkTokenID  = 100
)

type Tokenizer struct {
	vocab    map[string]int32
	vocabIDs map[int32]string
}

func NewTokenizer(vocabPath string) (*Tokenizer, error) {
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	t := &Tokenizer{
		vocab:    make(map[string]int32),
		vocabIDs: make(map[int32]string),
	}
	scanner := bufio.NewScanner(f)
	var id int32
	for scanner.Scan() {
		token := scanner.Text()
		t.vocab[token] = id
		t.vocabIDs[id] = token
		id++
	}
	return t, scanner.Err()
}

func (t *Tokenizer) VocabSize() int {
	return len(t.vocab)
}

func (t *Tokenizer) TokenToID(token string) (int32, bool) {
	id, ok := t.vocab[token]
	return id, ok
}

func (t *Tokenizer) IDToToken(id int32) string {
	return t.vocabIDs[id]
}

type TokenizedInput struct {
	InputIDs      []int64
	AttentionMask []int64
	TokenTypeIDs  []int64
}

func (t *Tokenizer) Encode(text string) TokenizedInput {
	tokens := t.tokenize(text)

	if len(tokens) > maxSeqLen-2 {
		tokens = tokens[:maxSeqLen-2]
	}

	inputIDs := make([]int64, 0, len(tokens)+2)
	attentionMask := make([]int64, 0, len(tokens)+2)
	tokenTypeIDs := make([]int64, 0, len(tokens)+2)

	inputIDs = append(inputIDs, clsTokenID)
	attentionMask = append(attentionMask, 1)
	tokenTypeIDs = append(tokenTypeIDs, 0)

	for _, token := range tokens {
		id, ok := t.vocab[token]
		if !ok {
			id = unkTokenID
		}
		inputIDs = append(inputIDs, int64(id))
		attentionMask = append(attentionMask, 1)
		tokenTypeIDs = append(tokenTypeIDs, 0)
	}

	inputIDs = append(inputIDs, sepTokenID)
	attentionMask = append(attentionMask, 1)
	tokenTypeIDs = append(tokenTypeIDs, 0)

	return TokenizedInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
		TokenTypeIDs:  tokenTypeIDs,
	}
}

func (t *Tokenizer) tokenize(text string) []string {
	text = basicClean(text)
	text = strings.ToLower(text)
	text = stripAccents(text)
	text = splitCJK(text)
	words := whitespaceTokenize(text)

	var result []string
	for _, word := range words {
		punctText := splitPunctuation(word)
		for _, tok := range whitespaceTokenize(punctText) {
			subs := t.wordPiece(tok)
			result = append(result, subs...)
		}
	}
	return result
}

func basicClean(text string) string {
	var buf strings.Builder
	for _, r := range text {
		if r == 0 || r == 0xfffd || unicode.IsControl(r) {
			if r == '\t' || r == '\n' || r == '\r' {
				buf.WriteRune(' ')
			}
			continue
		}
		buf.WriteRune(r)
	}
	return buf.String()
}

func stripAccents(text string) string {
	t := transform.Chain(norm.NFD, runes.Remove(runes.In(unicode.Mn)))
	result, _, _ := transform.String(t, text)
	return result
}

func whitespaceTokenize(text string) []string {
	return strings.Fields(text)
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		(unicode.Is(unicode.Hangul, r) && r >= 0xAC00 && r <= 0xD7A3)
}

func splitCJK(text string) string {
	var buf strings.Builder
	for _, r := range text {
		if isCJK(r) {
			buf.WriteRune(' ')
			buf.WriteRune(r)
			buf.WriteRune(' ')
		} else {
			buf.WriteRune(r)
		}
	}
	return buf.String()
}

func isPunctuation(r rune) bool {
	if r == '-' || r == '_' || r == '.' || r == '/' || r == '\'' {
		return false
	}
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

func (t *Tokenizer) wordPiece(word string) []string {
	if _, ok := t.vocab[word]; ok {
		return []string{word}
	}

	runes := []rune(word)
	start := 0
	var result []string

	for start < len(runes) {
		end := len(runes)
		var found string
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if _, ok := t.vocab[sub]; ok {
				found = sub
				break
			}
			end--
		}
		if found != "" {
			result = append(result, found)
			start = end
		} else {
			result = append(result, tokenUNK)
			start++
		}
	}
	return result
}

func splitPunctuation(text string) string {
	var buf strings.Builder
	runes := []rune(text)
	for i, r := range runes {
		prevIsPunct := i > 0 && isPunctuation(runes[i-1])
		nextIsPunct := i < len(runes)-1 && isPunctuation(runes[i+1])
		if isPunctuation(r) && (!prevIsPunct || !nextIsPunct) {
			buf.WriteRune(' ')
			buf.WriteRune(r)
			buf.WriteRune(' ')
		} else {
			buf.WriteRune(r)
		}
	}
	return buf.String()
}
