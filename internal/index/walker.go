package index

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

var defaultExcludes = []string{
	".git", ".svn", ".hg",
	"node_modules", "__pycache__", ".pytest_cache",
	"vendor", ".idea", ".vscode",
	".DS_Store", "Thumbs.db",
	"target", "build", "dist",
	".terraform", ".tox", ".mypy_cache", ".ruff_cache",
}

type WalkEntry struct {
	Path  string
	Info  fs.FileInfo
	RelPath string
}

type Walker struct {
	root        string
	includes    []string
	excludes    []string
	ignoreHidden bool
}

func NewWalker(root string) *Walker {
	return &Walker{
		root:        root,
		ignoreHidden: true,
	}
}

func (w *Walker) SetIncludes(patterns []string) {
	w.includes = patterns
}

func (w *Walker) SetExcludes(patterns []string) {
	w.excludes = patterns
}

func (w *Walker) Walk() ([]WalkEntry, error) {
	var entries []WalkEntry

	err := filepath.WalkDir(w.root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}

		if d.IsDir() {
			name := d.Name()
			if w.ignoreHidden && strings.HasPrefix(name, ".") && name != "." {
				return filepath.SkipDir
			}
			for _, ex := range defaultExcludes {
				if name == ex {
					return filepath.SkipDir
				}
			}
			return nil
		}

		if w.ignoreHidden && strings.HasPrefix(d.Name(), ".") {
			return nil
		}

		rel, _ := filepath.Rel(w.root, path)

		if len(w.excludes) > 0 && matchesAny(rel, w.excludes) {
			return nil
		}

		if len(w.includes) > 0 && !matchesAny(rel, w.includes) {
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}

		entries = append(entries, WalkEntry{
			Path:    path,
			Info:    info,
			RelPath: rel,
		})
		return nil
	})

	return entries, err
}

func IsTextFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	textExts := map[string]bool{
		".txt": true, ".md": true, ".rst": true, ".org": true,
		".go": true, ".py": true, ".rs": true, ".js": true, ".ts": true,
		".c": true, ".cpp": true, ".cc": true, ".cxx": true, ".h": true, ".hpp": true,
		".java": true, ".rb": true, ".php": true, ".swift": true, ".kt": true,
		".scala": true, ".clj": true, ".cljs": true, ".edn": true,
		".lua": true, ".r": true, ".jl": true, ".dart": true,
		".sh": true, ".bash": true, ".zsh": true, ".fish": true,
		".html": true, ".htm": true, ".css": true, ".scss": true, ".less": true,
		".json": true, ".xml": true, ".yaml": true, ".yml": true, ".toml": true,
		".csv": true, ".tsv": true, ".log": true,
		".cfg": true, ".conf": true, ".ini": true, ".env": true,
		".sql": true, ".proto": true, ".graphql": true,
		".tex": true, ".bib": true,
		".cmake": true, ".makefile": true, ".dockerfile": true,
		".tf": true, ".tfvars": true,
		".nim": true, ".zig": true, ".elm": true, ".hs": true, ".erl": true,
	}
	return textExts[ext]
}

func IsPDF(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".pdf"
}

func matchesAny(name string, patterns []string) bool {
	for _, p := range patterns {
		if matched, _ := filepath.Match(p, name); matched {
			return true
		}
	}
	return false
}

func ReadFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
