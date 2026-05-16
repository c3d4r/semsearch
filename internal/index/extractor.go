package index

import (
	"bytes"
	"fmt"
	"html"
	"regexp"
	"strings"

	"github.com/ledongthuc/pdf"
)

var htmlTagRE = regexp.MustCompile(`<[^>]*>`)

func ExtractText(path string) (string, error) {
	if IsPDF(path) {
		return extractPDF(path)
	}

	text, err := ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file %s: %w", path, err)
	}

	if isHTML(path) {
		text = stripHTML(text)
	}

	return text, nil
}

func isHTML(path string) bool {
	ext := strings.ToLower(path)
	return strings.HasSuffix(ext, ".html") || strings.HasSuffix(ext, ".htm")
}

func stripHTML(text string) string {
	text = htmlTagRE.ReplaceAllString(text, " ")
	text = html.UnescapeString(text)
	return text
}

func extractPDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if err != nil {
		return "", fmt.Errorf("open PDF %s: %w", path, err)
	}
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("read PDF %s: %w", path, err)
	}
	buf.ReadFrom(b)
	return buf.String(), nil
}
