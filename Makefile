.PHONY: build test lint clean setup release

BINARY := semsearch
PKG    := github.com/c3d4r/semsearch
GOOS   ?= linux
GOARCH ?= amd64

build:
	go build -o $(BINARY) ./cmd/$(BINARY)/

test:
	go test ./internal/... -v -count=1

test-short:
	go test ./internal/... -short -count=1

lint:
	go vet ./...

clean:
	rm -f $(BINARY)

setup:
	python3 scripts/export_model.py

release:
	GOOS=linux   GOARCH=amd64 go build -o dist/semsearch-linux-amd64   ./cmd/semsearch/
	GOOS=linux   GOARCH=arm64 go build -o dist/semsearch-linux-arm64   ./cmd/semsearch/
	GOOS=darwin  GOARCH=amd64 go build -o dist/semsearch-darwin-amd64  ./cmd/semsearch/
	GOOS=darwin  GOARCH=arm64 go build -o dist/semsearch-darwin-arm64  ./cmd/semsearch/
