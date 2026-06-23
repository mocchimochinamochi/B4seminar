package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// loadEnvFile は .env ファイルを読み込み，KEY=VALUE のマップを返す．
// python-dotenv 相当の最小実装（コメント行・空行は無視，前後の引用符は除去）．
func loadEnvFile(path string) (map[string]string, error) {
	result := map[string]string{}

	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return result, nil // .env が無い場合は空マップ（環境変数側で設定されている可能性もある）
		}
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.Index(line, "=")
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		val := strings.TrimSpace(line[idx+1:])
		val = strings.Trim(val, `"'`)
		result[key] = val
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// getAPIKeys は .env と実環境変数の両方から GEMINI_API_KEYS（カンマ区切り）を取得する．
// 実環境変数があればそれを優先する（python-dotenv の load_dotenv のデフォルト挙動に合わせる:
// 既存の環境変数は上書きしない＝環境変数優先）．
func getAPIKeys(envPath string) ([]string, error) {
	fileEnv, err := loadEnvFile(envPath)
	if err != nil {
		return nil, fmt.Errorf(".envの読み込みに失敗: %w", err)
	}

	raw := os.Getenv("GEMINI_API_KEYS")
	if raw == "" {
		raw = fileEnv["GEMINI_API_KEYS"]
	}

	var keys []string
	for _, k := range strings.Split(raw, ",") {
		k = strings.TrimSpace(k)
		if k != "" {
			keys = append(keys, k)
		}
	}
	return keys, nil
}
