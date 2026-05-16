package index

import (
	"testing"
)

func TestSplitterSmall(t *testing.T) {
	s := NewSplitter()
	chunks := s.Split("hello world")
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Text != "hello world" {
		t.Errorf("unexpected chunk text: %q", chunks[0].Text)
	}
}

func TestSplitterEmpty(t *testing.T) {
	s := NewSplitter()
	chunks := s.Split("")
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks, got %d", len(chunks))
	}
}

func TestSplitterMultiChunk(t *testing.T) {
	s := NewSplitter()
	s.ChunkSize = 50

	var text string
	for i := 0; i < 20; i++ {
		text += "This is a test sentence. "
	}

	chunks := s.Split(text)
	if len(chunks) < 2 {
		t.Errorf("expected multiple chunks, got %d", len(chunks))
	}

	for i, ch := range chunks {
		if ch.Text == "" {
			t.Errorf("chunk %d is empty", i)
		}
		if ch.Idx != i {
			t.Errorf("chunk %d has idx %d", i, ch.Idx)
		}
	}
}

func TestSplitterOverlap(t *testing.T) {
	s := NewSplitter()
	s.ChunkSize = 30
	s.ChunkOverlap = 10

	text := "This is sentence one. This is sentence two. This is sentence three. This is sentence four."

	chunks := s.Split(text)
	if len(chunks) < 2 {
		t.Errorf("expected at least 2 chunks, got %d", len(chunks))
	}
}

func TestIsTextFile(t *testing.T) {
	tests := []struct {
		path     string
		expected bool
	}{
		{"test.go", true},
		{"test.py", true},
		{"test.md", true},
		{"test.txt", true},
		{"test.pdf", false},
		{"test.png", false},
		{"test.zip", false},
		{"test", false},
		{"Makefile", false},
	}

	for _, tt := range tests {
		got := IsTextFile(tt.path)
		if got != tt.expected {
			t.Errorf("IsTextFile(%q) = %v, want %v", tt.path, got, tt.expected)
		}
	}
}

func TestIsPDF(t *testing.T) {
	mustBePDF := []string{"test.pdf", "test.PDF", "/path/to/doc.Pdf"}
	for _, p := range mustBePDF {
		if !IsPDF(p) {
			t.Errorf("IsPDF(%q) should be true", p)
		}
	}
	notPDF := []string{"test.txt", "test.pdf.txt", "test"}
	for _, p := range notPDF {
		if IsPDF(p) {
			t.Errorf("IsPDF(%q) should be false", p)
		}
	}
}

func TestStripHTML(t *testing.T) {
	html := "<p>Hello <b>World</b></p>"
	got := stripHTML(html)
	if !contains(got, "Hello") || !contains(got, "World") {
		t.Errorf("stripHTML: got %q, expected it to contain Hello and World", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestSentenceSplit(t *testing.T) {
	sentences := splitSentences("Hello world. This is a test! What about questions? And more...")
	if len(sentences) < 3 {
		t.Errorf("expected at least 3 sentences, got %d: %v", len(sentences), sentences)
	}
}
