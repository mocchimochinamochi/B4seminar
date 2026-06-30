package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// geminiEndpointFmt はGemini APIのエンドポイント書式。テストでは差し替えてモックサーバーを指す。
var geminiEndpointFmt = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s"

type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type generationConfig struct {
	Temperature float64 `json:"temperature"`
}

type geminiRequest struct {
	Contents         []geminiContent  `json:"contents"`
	GenerationConfig generationConfig `json:"generationConfig"`
}

type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

type geminiAPIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
	Error      *geminiAPIError   `json:"error,omitempty"`
}

// rateLimitError は429(レート制限)を表すエラー型．呼び出し側でAPIキー切替の判断に使う．
type rateLimitError struct {
	msg string
}

func (e *rateLimitError) Error() string { return e.msg }

// callGemini はGemini APIにプロンプトを送り，テキスト応答を返す．
func callGemini(ctx context.Context, httpClient *http.Client, model, apiKey, prompt string, temperature float64) (string, error) {
	reqBody := geminiRequest{
		Contents:         []geminiContent{{Parts: []geminiPart{{Text: prompt}}}},
		GenerationConfig: generationConfig{Temperature: temperature},
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("リクエストのJSON化に失敗: %w", err)
	}

	url := fmt.Sprintf(geminiEndpointFmt, model, apiKey)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return "", fmt.Errorf("リクエスト生成に失敗: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTPリクエストに失敗: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("レスポンス読み取りに失敗: %w", err)
	}

	if resp.StatusCode == http.StatusTooManyRequests {
		return "", &rateLimitError{msg: fmt.Sprintf("429 Too Many Requests: %s", string(body))}
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("HTTPステータス %d: %s", resp.StatusCode, string(body))
	}

	var parsed geminiResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("レスポンスのJSON解析に失敗: %w (body=%s)", err, string(body))
	}
	if parsed.Error != nil {
		if parsed.Error.Code == http.StatusTooManyRequests {
			return "", &rateLimitError{msg: parsed.Error.Message}
		}
		return "", fmt.Errorf("APIエラー: %s", parsed.Error.Message)
	}
	if len(parsed.Candidates) == 0 {
		return "", nil
	}

	var sb strings.Builder
	for _, part := range parsed.Candidates[0].Content.Parts {
		sb.WriteString(part.Text)
	}
	return strings.TrimSpace(sb.String()), nil
}

// httpClientWithTimeout は適度なタイムアウトを設定したHTTPクライアントを返す．
func httpClientWithTimeout(d time.Duration) *http.Client {
	return &http.Client{Timeout: d}
}
