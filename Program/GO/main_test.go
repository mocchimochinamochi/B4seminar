package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestBuildTrials(t *testing.T) {
	for _, n := range []int{100, 50, 1, 0} {
		trials := buildTrials(n)
		if len(trials) != n {
			t.Fatalf("buildTrials(%d) = %d trials, want %d", n, len(trials), n)
		}
		for i, tr := range trials {
			if tr.Index != i {
				t.Fatalf("trial %d has Index=%d, want %d", i, tr.Index, i)
			}
		}
	}
}

func TestExtractNumbers(t *testing.T) {
	cases := []struct {
		in   string
		want []int
	}{
		{"1,2,3,4,5,6,7,8,9,10,0,1,2,3,4,5,6,7,8", []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8}},
		{"回答: 5, 6, 7", []int{5, 6, 7}},
		{"", nil},
	}
	for _, c := range cases {
		got := extractNumbers(c.in)
		if len(got) != len(c.want) {
			t.Fatalf("extractNumbers(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("extractNumbers(%q)[%d] = %d, want %d", c.in, i, got[i], c.want[i])
			}
		}
	}
}

// mockGeminiResponse builds a minimal valid Gemini API JSON response containing the given text.
func mockGeminiResponse(text string) []byte {
	resp := geminiResponse{
		Candidates: []geminiCandidate{{Content: geminiContent{Parts: []geminiPart{{Text: text}}}}},
	}
	b, _ := json.Marshal(resp)
	return b
}

// TestParallelProcessingAgainstMockServer spins up several "API keys" (each mapped to a counter)
// behind a mock HTTP server, runs the full worker-pool pipeline, and checks that:
//   - every trial ends up with exactly 19 parsed values,
//   - results are written safely to CSV with no corruption despite concurrent workers,
//   - a simulated 429 on one key causes rotation to another key rather than failing the trial.
func TestParallelProcessingAgainstMockServer(t *testing.T) {
	const n = 24
	const numKeys = 4

	var callCounts [numKeys]int64
	var rateLimitedOnce sync.Once
	rateLimitTripped := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := r.URL.Query().Get("key")
		var keyIdx int
		fmt.Sscanf(key, "key%d", &keyIdx)
		if keyIdx >= 0 && keyIdx < numKeys {
			atomic.AddInt64(&callCounts[keyIdx], 1)
		}

		// Simulate exactly one rate-limit event on the very first key to exercise rotation.
		limitedThisCall := false
		if keyIdx == 0 {
			rateLimitedOnce.Do(func() {
				rateLimitTripped = true
				limitedThisCall = true
				w.WriteHeader(http.StatusTooManyRequests)
				w.Write([]byte(`{"error":{"code":429,"message":"rate limited","status":"RESOURCE_EXHAUSTED"}}`))
			})
		}
		if limitedThisCall {
			return
		}

		text := "1,2,3,4,5,6,7,8,9,10,0,1,2,3,4,5,6,7,8"
		w.WriteHeader(http.StatusOK)
		w.Write(mockGeminiResponse(text))
	}))
	defer server.Close()

	// Point the client at the mock server instead of the real Gemini endpoint.
	oldFmt := geminiEndpointFmt
	geminiEndpointFmt = server.URL + "/%s?key=%s"
	defer func() { geminiEndpointFmt = oldFmt }()

	keys := make([]string, numKeys)
	for i := range keys {
		keys[i] = fmt.Sprintf("key%d", i)
	}
	km := NewKeyManager(keys)

	dir := t.TempDir()
	csvPath := filepath.Join(dir, "results.csv")
	if err := ensureCSVHeader(csvPath); err != nil {
		t.Fatalf("ensureCSVHeader: %v", err)
	}

	trials := buildTrials(n)
	jobs := make(chan Trial, len(trials))
	for _, tr := range trials {
		jobs <- tr
	}
	close(jobs)

	results := make(chan TrialResult, n)
	httpClient := httpClientWithTimeout(5 * time.Second)

	const workers = 6
	var wg sync.WaitGroup
	ctx := context.Background()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for tr := range jobs {
				values, err := processTrial(ctx, httpClient, km, "test-model", tr, 5)
				if err != nil {
					t.Errorf("trial %d failed: %v", tr.Index, err)
					continue
				}
				results <- TrialResult{Index: tr.Index, Persona: tr.Persona, Values: values}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	runResultWriter(csvPath, results, n)

	completed, err := readCompletedIndices(csvPath)
	if err != nil {
		t.Fatalf("readCompletedIndices: %v", err)
	}
	if len(completed) != n {
		t.Fatalf("got %d completed trials, want %d (possible CSV race/corruption)", len(completed), n)
	}
	for i := 0; i < n; i++ {
		if !completed[i] {
			t.Fatalf("trial %d missing from CSV", i)
		}
	}

	rows, err := readAllValueRows(csvPath)
	if err != nil {
		t.Fatalf("readAllValueRows: %v", err)
	}
	if len(rows) != n {
		t.Fatalf("got %d value rows, want %d", len(rows), n)
	}
	for _, row := range rows {
		if len(row) != NumQuestions {
			t.Fatalf("row has %d values, want %d", len(row), NumQuestions)
		}
	}

	if !rateLimitTripped {
		t.Fatalf("expected the mock 429 to have been triggered at least once")
	}

	totalCalls := int64(0)
	for _, c := range callCounts {
		totalCalls += c
	}
	t.Logf("calls per key: %v (total %d, trials %d)", callCounts, totalCalls, n)
	if totalCalls < int64(n) {
		t.Fatalf("expected at least %d calls across keys, got %d", n, totalCalls)
	}
}

func TestGeneratePlotsWritesSVGFiles(t *testing.T) {
	dir := t.TempDir()
	rows := [][]int{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8},
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8},
	}
	if err := generatePlots(rows, dir); err != nil {
		t.Fatalf("generatePlots: %v", err)
	}
	for q := 1; q <= NumQuestions; q++ {
		p := filepath.Join(dir, fmt.Sprintf("question_%d.svg", q))
		data, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("missing svg for question %d: %v", q, err)
		}
		if !strings.HasPrefix(string(data), "<svg") {
			t.Fatalf("question_%d.svg does not start with <svg", q)
		}
	}
}
