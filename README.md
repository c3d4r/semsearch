# semsearch

Semantic filesystem search in Go. Index a directory tree and query it with natural language.

Uses `all-MiniLM-L6-v2` via ONNX Runtime for embeddings and HNSW for fast approximate nearest-neighbor search. Everything runs locally — no server, no API key, no telemetry.

## Quickstart

```bash
# 1. Install ONNX Runtime (system dependency)
# Ubuntu/Debian:
sudo apt install libonnxruntime
# macOS:
brew install onnxruntime
# Or download from: https://github.com/microsoft/onnxruntime/releases

# 2. Export the embedding model (one-time, requires Python)
pip install transformers torch
python3 scripts/export_model.py
# Creates models/model.onnx (~80MB) and models/vocab.txt

# 3. Build
make build

# 4. Index a directory
./semsearch index ~/my-project

# 5. Search
./semsearch search "how does authentication work"
```

## Commands

```
semsearch index [dir]        Build a search index over a directory
    -i, --include <glob>     File patterns to include (repeatable)
    -e, --exclude <glob>     File patterns to exclude
    -f, --force              Force re-index even if unchanged
    --index-dir <path>       Store index at path (default: .semsearch/)

semsearch search <query>     Semantic search over the index
    -n, --num <int>          Number of results (default: 10)
    --no-snippet             Hide snippet preview
    --index-dir <path>       Path to index directory

semsearch status             Show index stats (files, chunks, size)
semsearch clear              Remove the index
```

## Supported file types

- Text: `.txt`, `.md`, `.rst`, `.org`
- Code: `.go`, `.py`, `.rs`, `.js`, `.ts`, `.c`, `.h`, `.java`, `.rb`, plus 30+ more
- Documents: `.pdf` (basic text extraction)
- Config: `.json`, `.yaml`, `.toml`, `.xml`, `.csv`, `.log`
- HTML: `.html`, `.htm` (tags stripped)

## How it works

```
Walk directory tree
  → Extract text (plain read, PDF via ledongthuc/pdf, HTML via tag stripping)
  → Chunk text (~1500 chars, sentence-aware, 25% overlap)
  → Tokenize with WordPiece (BERT vocab, pure Go)
  → Embed with all-MiniLM-L6-v2 via ONNX Runtime (384-dim vectors)
  → Store vectors in HNSW index (cosine distance, pure Go)
  → Store metadata in SQLite (chunk text, file paths, hashes)
```

Incremental re-indexing: file content is hashed (xxhash64). On subsequent runs, unchanged files are skipped. Changed files have old chunks removed and new ones added.

## Index storage

Everything lives in `.semsearch/` at the root of the indexed directory:
- `index.db` — SQLite database (chunk text, file metadata)
- `index.hnsw` — HNSW vector index (binary format)

Model files are cached in `~/.cache/semsearch/models/`.

## Architecture

```
cmd/semsearch/          CLI entry point (cobra)
internal/
  embed/                ONNX embedding engine + WordPiece tokenizer
    tokenizer.go        Pure Go BERT WordPiece tokenizer
    onnx.go             ONNX Runtime wrapper
    pool.go             Mean pooling + L2 normalization
  store/                Persistence layer
    sqlite.go           Chunk metadata + file index state
    hnsw.go             HNSW vector index (coder/hnsw)
  index/                Indexing pipeline
    walker.go           File tree walker with glob/ignore filters
    extractor.go        Text extraction (plain, PDF, HTML)
    chunker.go          Sentence-aware sliding window chunker
    index.go            Orchestration
  search/               Query processing
    search.go           Embed query → ANN → lookup → format
  model/                Model provisioning
    download.go         Auto-download from GitHub Releases
scripts/
  export_model.py       Export all-MiniLM-L6-v2 to ONNX
```

## Testing

```bash
go test ./... -v
```

## Requirements

- Go 1.21+
- libonnxruntime (system package or manual install)
- Python 3.9+ (for `scripts/export_model.py` only)

## License

MIT
