package model

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

const (
	defaultBaseURL  = "https://github.com/c3d4r/semsearch/releases/download/v0.1.0"
	modelFileName   = "model.onnx"
	vocabFileName   = "vocab.txt"
	cacheDirName    = "semsearch"
	modelsDirName   = "models"
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
	dir, err := ModelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, modelFileName), nil
}

func VocabPath() (string, error) {
	dir, err := ModelsDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, vocabFileName), nil
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
	localVocab := filepath.Join("models", vocabFileName)
	if _, err := os.Stat(localModel); err == nil {
		if _, err := os.Stat(localVocab); err == nil {
			fmt.Println("Using local model files from ./models/")
			return copyFiles(localModel, localVocab, modelPath, vocabPath)
		}
	}

	baseURL := os.Getenv("SEMSEARCH_MODEL_URL")
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Downloading %s...\n", modelFileName)
		if err := downloadFile(modelPath, baseURL+"/"+modelFileName); err != nil {
			return fmt.Errorf("download model: %w (place model.onnx and vocab.txt in %s, or set SEMSEARCH_MODEL_URL)", err, modelsDir)
		}
	}

	if _, err := os.Stat(vocabPath); os.IsNotExist(err) {
		fmt.Printf("Downloading %s...\n", vocabFileName)
		if err := downloadFile(vocabPath, baseURL+"/"+vocabFileName); err != nil {
			return fmt.Errorf("download vocab: %w", err)
		}
	}

	return nil
}

func copyFiles(srcModel, srcVocab, dstModel, dstVocab string) error {
	for _, pair := range [][2]string{{srcModel, dstModel}, {srcVocab, dstVocab}} {
		src, err := os.Open(pair[0])
		if err != nil {
			return err
		}
		defer src.Close()
		dst, err := os.Create(pair[1])
		if err != nil {
			return err
		}
		defer dst.Close()
		if _, err := io.Copy(dst, src); err != nil {
			return err
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

	switch runtime.GOOS {
	case "linux":
		candidates := []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				return c, nil
			}
		}
		return "", fmt.Errorf(
			"libonnxruntime.so not found. Install it or set ONNXRUNTIME_LIB env var.\n" +
				"  Ubuntu/Debian: sudo apt install libonnxruntime\n" +
				"  Or download from: https://github.com/microsoft/onnxruntime/releases")
	case "darwin":
		candidates := []string{
			"/usr/local/lib/libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				return c, nil
			}
		}
		return "", fmt.Errorf("libonnxruntime.dylib not found. Install it or set ONNXRUNTIME_LIB env var")
	default:
		return "", fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}
}
