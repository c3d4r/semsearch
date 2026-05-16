package model

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	defaultBaseURL     = "https://github.com/c3d4r/semsearch/releases/download/v0.1.0"
	modelFileName      = "model.onnx"
	vocabFileName      = "vocab.txt"
	cacheDirName       = "semsearch"
	modelsDirName      = "models"
	libDirName         = "lib"
	ortDownloadVersion = "1.25.0"
)

func CacheDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}

	cache := os.Getenv("XDG_CACHE_HOME")
	if cache == "" {
		cache = filepath.Join(home, ".cache")
	}

	return filepath.Join(cache, cacheDirName), nil
}

func ModelsDir() (string, error) {
	cache, err := CacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(cache, modelsDirName), nil
}

func ModelPath() (string, error) {
	local := filepath.Join("models", modelFileName)
	if _, err := os.Stat(local); err == nil {
		return local, nil
	}
	dir, err := ModelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, modelFileName), nil
}

func VocabPath() (string, error) {
	local := filepath.Join("models", vocabFileName)
	if _, err := os.Stat(local); err == nil {
		return local, nil
	}
	dir, err := ModelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, vocabFileName), nil
}

func LibDir() (string, error) {
	cache, err := CacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(cache, libDirName), nil
}

func EnsureModels() error {
	modelsDir, err := ModelsDir()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(modelsDir, 0755); err != nil {
		return fmt.Errorf("create models dir: %w", err)
	}

	modelPath := filepath.Join(modelsDir, modelFileName)
	vocabPath := filepath.Join(modelsDir, vocabFileName)

	if _, err := os.Stat(modelPath); err == nil {
		if _, err := os.Stat(vocabPath); err == nil {
			return nil
		}
	}

	localModel := filepath.Join("models", modelFileName)
	if _, err := os.Stat(localModel); err == nil {
		fmt.Println("Using local model files from ./models/")
		modelsDir = "models"
		return nil
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		baseURL := os.Getenv("SEMSEARCH_MODEL_URL")
		if baseURL == "" {
			baseURL = defaultBaseURL
		}
		fmt.Printf("Downloading %s...\n", modelFileName)
		if err := downloadFile(modelPath, baseURL+"/"+modelFileName); err != nil {
			return fmt.Errorf("download model: %w", err)
		}
	}

	if _, err := os.Stat(vocabPath); os.IsNotExist(err) {
		baseURL := os.Getenv("SEMSEARCH_MODEL_URL")
		if baseURL == "" {
			baseURL = defaultBaseURL
		}
		fmt.Printf("Downloading %s...\n", vocabFileName)
		if err := downloadFile(vocabPath, baseURL+"/"+vocabFileName); err != nil {
			return fmt.Errorf("download vocab: %w", err)
		}
	}

	return nil
}

func downloadFile(dest, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, url)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}

func FindONNXLibrary() (string, error) {
	lib := os.Getenv("ONNXRUNTIME_LIB")
	if lib != "" {
		if _, err := os.Stat(lib); err == nil {
			return lib, nil
		}
	}

	if p := findInCacheDir(); p != "" {
		return p, nil
	}

	for _, c := range onnxLibCandidates() {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	return EnsureONNXLib()
}

func findInCacheDir() string {
	cache, err := CacheDir()
	if err != nil {
		return ""
	}
	libDir := filepath.Join(cache, libDirName)
	entries, err := os.ReadDir(libDir)
	if err != nil {
		return ""
	}
	prefix := "libonnxruntime"
	suffix := ".so"
	if runtime.GOOS == "darwin" {
		suffix = ".dylib"
	}
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, suffix) {
			return filepath.Join(libDir, name)
		}
	}
	return ""
}

func onnxLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

func onnxLibCandidates() []string {
	libName := onnxLibName()

	candidates := []string{}

	home, err := os.UserHomeDir()
	if err == nil {
		candidates = append(candidates,
			filepath.Join(home, ".local", "lib", libName),
			filepath.Join(home, ".cache", "semsearch", "lib", libName),
		)
	}

	switch runtime.GOOS {
	case "linux":
		candidates = append(candidates,
			"/usr/lib/"+libName,
			"/usr/lib/x86_64-linux-gnu/"+libName,
			"/usr/local/lib/"+libName,
		)
	case "darwin":
		candidates = append(candidates,
			"/usr/local/lib/"+libName,
			"/opt/homebrew/lib/"+libName,
		)
	}

	return candidates
}

func EnsureONNXLib() (string, error) {
	libDir, err := LibDir()
	if err != nil {
		return "", err
	}

	if err := os.MkdirAll(libDir, 0755); err != nil {
		return "", fmt.Errorf("create lib dir: %w", err)
	}

	entries, _ := os.ReadDir(libDir)
	for _, e := range entries {
		name := e.Name()
		prefix := "libonnxruntime"
		suffix := ".so"
		if runtime.GOOS == "darwin" {
			suffix = ".dylib"
		}
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, suffix) {
			return filepath.Join(libDir, name), nil
		}
	}

	url := ortDownloadURL()
	if url == "" {
		return "", fmt.Errorf("no ONNX Runtime download available for %s/%s — install it manually: https://github.com/microsoft/onnxruntime/releases", runtime.GOOS, runtime.GOARCH)
	}

	fmt.Printf("Downloading ONNX Runtime %s for %s/%s...\n", ortDownloadVersion, runtime.GOOS, runtime.GOARCH)
	if err := downloadAndExtractONNX(url, libDir); err != nil {
		return "", fmt.Errorf("download ONNX Runtime: %w", err)
	}

	entries, _ = os.ReadDir(libDir)
	for _, e := range entries {
		name := e.Name()
		prefix := "libonnxruntime"
		suffix := ".so"
		if runtime.GOOS == "darwin" {
			suffix = ".dylib"
		}
		if strings.HasPrefix(name, prefix) && strings.HasSuffix(name, suffix) {
			return filepath.Join(libDir, name), nil
		}
	}

	return "", fmt.Errorf("ONNX Runtime not found after download in %s", libDir)
}

func ortDownloadURL() string {
	base := fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s", ortDownloadVersion)
	switch runtime.GOOS {
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return base + "/onnxruntime-linux-x64-" + ortDownloadVersion + ".tgz"
		case "arm64":
			return base + "/onnxruntime-linux-aarch64-" + ortDownloadVersion + ".tgz"
		}
	case "darwin":
		switch runtime.GOARCH {
		case "amd64":
			return base + "/onnxruntime-osx-x86_64-" + ortDownloadVersion + ".tgz"
		case "arm64":
			return base + "/onnxruntime-osx-arm64-" + ortDownloadVersion + ".tgz"
		}
	}
	return ""
}

func downloadAndExtractONNX(url, destDir string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, url)
	}

	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gzReader.Close()

	tarReader := tar.NewReader(gzReader)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		name := filepath.Base(header.Name)
		if !strings.HasPrefix(name, "libonnxruntime") {
			continue
		}
		suffix := ".so"
		if runtime.GOOS == "darwin" {
			suffix = ".dylib"
		}
		if !strings.HasSuffix(name, suffix) {
			continue
		}

		destPath := filepath.Join(destDir, name)
		f, err := os.OpenFile(destPath, os.O_CREATE|os.O_WRONLY, 0755)
		if err != nil {
			return err
		}
		defer f.Close()

		if _, err := io.Copy(f, tarReader); err != nil {
			return err
		}
		return nil
	}

	return fmt.Errorf("libonnxruntime not found in archive")
}
