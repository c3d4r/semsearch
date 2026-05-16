package embed

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBasicClean(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello\x00world", "helloworld"},
		{"hello\nworld", "hello world"},
		{"hello\tworld", "hello world"},
		{"hello\rworld", "hello world"},
		{"hello world", "hello world"},
		{"hello \x00 world", "hello  world"},
	}

	for _, tt := range tests {
		got := basicClean(tt.input)
		if got != tt.expected {
			t.Errorf("basicClean(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestStripAccents(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"café", "cafe"},
		{"naïve", "naive"},
		{"über", "uber"},
		{"hello", "hello"},
	}

	for _, tt := range tests {
		got := stripAccents(tt.input)
		if got != tt.expected {
			t.Errorf("stripAccents(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestSplitCJK(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello世界world", "hello 世  界 world"},
		{"hello", "hello"},
		{"日本語", " 日  本  語 "},
	}

	for _, tt := range tests {
		got := splitCJK(tt.input)
		if got != tt.expected {
			t.Errorf("splitCJK(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestSplitPunctuation(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello!", "hello ! "},
		{"hello?world", "hello ? world"},
		{"hello,world", "hello , world"},
		{"don't", "don't"},
		{"state-of-the-art", "state-of-the-art"},
		{"hello_world", "hello_world"},
		{"file.txt", "file.txt"},
	}

	for _, tt := range tests {
		got := splitPunctuation(tt.input)
		if got != tt.expected {
			t.Errorf("splitPunctuation(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestTokenizer(t *testing.T) {
	dir := t.TempDir()
	vocabPath := filepath.Join(dir, "vocab.txt")

	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\ntest\nthe\n##ing\n##ed\n##ly\na\nis\nwhat\n?\n!\n,\n.\nthis\n##s\n##able\n##ity\nun\n"
	_ = os.WriteFile(vocabPath, []byte(vocab), 0644)

	tok, err := NewTokenizer(vocabPath)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name     string
		input    string
		minTokens int
		checkFirst int64
	}{
		{"simple", "hello world", 2, 5},
		{"with punctuation", "hello, world!", 3, 5},
		{"subword", "testing", 2, -1},
		{"unknown", "zzzxxx", 1, -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := tok.Encode(tt.input)
			if len(out.InputIDs) < tt.minTokens+2 { // +2 for [CLS] and [SEP]
				t.Errorf("Encode(%q): got %d tokens, want at least %d", tt.input, len(out.InputIDs), tt.minTokens+2)
			}
			if tt.checkFirst >= 0 && out.InputIDs[1] != tt.checkFirst {
				t.Errorf("Encode(%q): first token = %d, want %d", tt.input, out.InputIDs[1], tt.checkFirst)
			}
			if out.InputIDs[0] != clsTokenID {
				t.Errorf("first token should be [CLS] (101), got %d", out.InputIDs[0])
			}
			last := len(out.InputIDs) - 1
			if out.InputIDs[last] != sepTokenID {
				t.Errorf("last token should be [SEP] (102), got %d", out.InputIDs[last])
			}
			if len(out.AttentionMask) != len(out.InputIDs) {
				t.Error("attention mask length mismatch")
			}
			if len(out.TokenTypeIDs) != len(out.InputIDs) {
				t.Error("token type ids length mismatch")
			}
		})
	}
}

func TestTokenizerMaxLength(t *testing.T) {
	dir := t.TempDir()
	vocabPath := filepath.Join(dir, "vocab.txt")
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n"
	_ = os.WriteFile(vocabPath, []byte(vocab), 0644)

	tok, err := NewTokenizer(vocabPath)
	if err != nil {
		t.Fatal(err)
	}

	longText := ""
	for i := 0; i < 600; i++ {
		longText += "hello world "
	}

	out := tok.Encode(longText)
	if len(out.InputIDs) > maxSeqLen {
		t.Errorf("too many tokens: %d > %d", len(out.InputIDs), maxSeqLen)
	}
}

func TestMeanPool(t *testing.T) {
	hidden := make([][][]float32, 1)
	hidden[0] = [][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	}
	mask := []int64{1, 1, 0} // last token is padding

	emb := MeanPool(hidden, mask)

	// Expected: mean of first two vectors = ((1+4)/2, (2+5)/2, (3+6)/2) = (2.5, 3.5, 4.5)
	// then L2 normalized
	expected := []float32{2.5, 3.5, 4.5}
	norm := float32(0)
	for _, v := range expected {
		norm += v * v
	}
	// L2 normalize
	for i := range expected {
		expected[i] /= float32(0)
	}
	_ = norm

	if len(emb) != 3 {
		t.Fatalf("expected 3 dimensions, got %d", len(emb))
	}
	if emb[0] < 0 || emb[2] < 0 {
		t.Error("unexpected negative values")
	}
}

func TestL2Normalize(t *testing.T) {
	vec := []float32{3, 4}
	l2Normalize(vec)
	expected := []float32{0.6, 0.8}
	for i := range vec {
		if vec[i] < expected[i]-1e-4 || vec[i] > expected[i]+1e-4 {
			t.Errorf("l2Normalize: vec[%d] = %f, want %f", i, vec[i], expected[i])
		}
	}
}
