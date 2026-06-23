// surveybot は，複数の性格ペルソナを与えたGemini APIへ繰り返しアンケートを投げかけ，
// 各設問の回答分布を集計・可視化するツール（元のPythonスクリプトのGo移植版）．
//
// 元のPython版との主な違い:
//   - 複数ワーカーによる並列実行に対応（-workers フラグ）。
//   - APIキーのローテーションはゴルーチン間で共有される KeyManager で安全に行う。
//   - CSVへの書き込みは単一の書き込みゴルーチンに集約し，競合を避ける。
//   - 中断・再実行時のレジューム判定は，trialインデックスの集合で行うため，
//     並列実行で完了順序が入れ替わっても安全に再開できる。
//   - matplotlibの代わりに，外部依存のないSVGバーチャートを生成する。
package main

import (
	"context"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// Trial は実行すべき1件のアンケート（インデックスとペルソナの組）．
type Trial struct {
	Index   int
	Persona Persona
}

// TrialResult は完了したTrialの回答結果．
type TrialResult struct {
	Index   int
	Persona Persona
	Values  []int
}

var numberRe = regexp.MustCompile(`\d+`)

func main() {
	var (
		nTotal     = flag.Int("n", 100, "総試行回数")
		workers    = flag.Int("workers", 0, "並列ワーカー数 (0 の場合はAPIキー数に合わせて自動設定)")
		model      = flag.String("model", "gemini-3-flash-preview", "使用するGeminiモデル名")
		csvPath    = flag.String("csv", "survey_results_persona.csv", "結果を保存するCSVファイルパス")
		plotsDir   = flag.String("plots", "plots", "グラフ(SVG)の出力先ディレクトリ")
		envPath    = flag.String("env", ".env", ".envファイルのパス")
		maxRetries = flag.Int("max-retries", 8, "1試行あたりの最大リトライ回数(レート制限による待機は除く)")
		timeoutSec = flag.Int("timeout", 60, "1回のAPIリクエストのタイムアウト(秒)")
	)
	flag.Parse()

	apiKeys, err := getAPIKeys(*envPath)
	if err != nil {
		log.Fatalf("エラー: %v", err)
	}
	if len(apiKeys) == 0 {
		log.Fatalf("エラー: %s または環境変数に GEMINI_API_KEYS が正しく設定されていません。", *envPath)
	}
	km := NewKeyManager(apiKeys)

	if *workers <= 0 {
		*workers = km.Len()
		if *workers < 1 {
			*workers = 1
		}
	}

	trials := buildTrials(*nTotal)

	if err := ensureCSVHeader(*csvPath); err != nil {
		log.Fatalf("エラー: CSVヘッダーの作成に失敗: %v", err)
	}

	completed, err := readCompletedIndices(*csvPath)
	if err != nil {
		log.Fatalf("エラー: 既存CSVの読み込みに失敗: %v", err)
	}

	fmt.Printf("Total trials to run: %d\n", len(trials))
	if len(completed) > 0 {
		fmt.Printf("Resuming: %d/%d trials already completed.\n", len(completed), len(trials))
	}
	fmt.Printf("Using %d parallel worker(s) across %d API key(s).\n", *workers, km.Len())

	// Ctrl+C で安全に中断できるようにする（実行中のリクエストは完了を待つ）。
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	jobs := make(chan Trial, len(trials))
	for _, t := range trials {
		if completed[t.Index] {
			continue
		}
		jobs <- t
	}
	close(jobs)

	results := make(chan TrialResult, *workers)
	httpClient := httpClientWithTimeout(time.Duration(*timeoutSec) * time.Second)

	var workersWG sync.WaitGroup
	for w := 0; w < *workers; w++ {
		workersWG.Add(1)
		go func(workerID int) {
			defer workersWG.Done()
			for trial := range jobs {
				select {
				case <-ctx.Done():
					return
				default:
				}
				values, err := processTrial(ctx, httpClient, km, *model, trial, *maxRetries)
				if err != nil {
					log.Printf("Trial %d: %d回試行後も失敗しました。スキップします (%v)", trial.Index, *maxRetries, err)
					continue
				}
				select {
				case results <- TrialResult{Index: trial.Index, Persona: trial.Persona, Values: values}:
				case <-ctx.Done():
					return
				}
			}
		}(w)
	}

	// 全ワーカー終了後に results チャネルを閉じる。
	go func() {
		workersWG.Wait()
		close(results)
	}()

	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		runResultWriter(*csvPath, results, len(trials))
	}()

	<-writerDone

	fmt.Println("グラフを生成・保存中...")
	rows, err := readAllValueRows(*csvPath)
	if err != nil {
		log.Fatalf("エラー: CSV再読み込みに失敗: %v", err)
	}
	if len(rows) == 0 {
		fmt.Println("結果が0件のため，グラフ生成をスキップしました。")
		return
	}
	if err := generatePlots(rows, *plotsDir); err != nil {
		log.Fatalf("エラー: グラフ生成に失敗: %v", err)
	}
	fmt.Printf("すべてのグラフを '%s' フォルダに保存しました。\n", *plotsDir)
}

// buildTrials は重み付けに基づき，総数 n 件のTrialリストを構築する。
// 元のPython版と同じロジック（round(weight*n)で配分し，不足分は先頭ペルソナで埋め，
// 過剰分は末尾から削る）。
func buildTrials(n int) []Trial {
	var trials []Trial
	for _, p := range PERSONAS {
		count := int(math.Round(p.Weight * float64(n)))
		for i := 0; i < count; i++ {
			trials = append(trials, Trial{Persona: p})
		}
	}
	for len(trials) < n {
		trials = append(trials, Trial{Persona: PERSONAS[0]})
	}
	for len(trials) > n {
		trials = trials[:len(trials)-1]
	}
	for i := range trials {
		trials[i].Index = i
	}
	return trials
}

func buildPrompt(p Persona) string {
	return fmt.Sprintf("あなたは%sの日本人です。性格タイプは%sです。\n%s", p.Desc, p.Type, QuestionText)
}

// processTrial は1件のTrialについて，パース可能な19個の数値が得られるまでリトライする。
// レート制限(429)はAPIキーをローテーションして即時リトライし，リトライ回数の上限には数えない。
func processTrial(ctx context.Context, httpClient *http.Client, km *KeyManager, model string, trial Trial, maxRetries int) ([]int, error) {
	prompt := buildPrompt(trial.Persona)

	attempts := 0
	for attempts < maxRetries {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		text, err := callGemini(ctx, httpClient, model, km.Current(), prompt, 1.0)
		if err != nil {
			var rle *rateLimitError
			if errors.As(err, &rle) {
				km.Rotate()
				log.Printf("Trial %d: レート制限を検知。APIキーを切り替えます", trial.Index)
				sleepCtx(ctx, 5*time.Second)
				continue // レート制限はリトライ回数に数えない
			}
			log.Printf("Trial %d: Error - %v", trial.Index, err)
			attempts++
			sleepCtx(ctx, 2*time.Second)
			continue
		}

		if text == "" {
			log.Printf("Trial %d: 空のレスポンスを受信 → リトライ", trial.Index)
			attempts++
			sleepCtx(ctx, 2*time.Second)
			continue
		}

		values := extractNumbers(text)
		if len(values) != NumQuestions {
			log.Printf("Trial %d: 回答数不一致 (%d個検出) → リトライ", trial.Index, len(values))
			attempts++
			sleepCtx(ctx, 1*time.Second)
			continue
		}

		return values, nil
	}
	return nil, fmt.Errorf("最大リトライ回数(%d)に到達", maxRetries)
}

func sleepCtx(ctx context.Context, d time.Duration) {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
	case <-ctx.Done():
	}
}

func extractNumbers(text string) []int {
	matches := numberRe.FindAllString(text, -1)
	values := make([]int, 0, len(matches))
	for _, m := range matches {
		v, err := strconv.Atoi(m)
		if err != nil {
			continue
		}
		values = append(values, v)
	}
	return values
}

// ---- CSV関連 ----

func csvHeader() []string {
	header := []string{"Trial", "PersonaType", "PersonaDesc"}
	for i := 1; i <= NumQuestions; i++ {
		header = append(header, fmt.Sprintf("Q%d", i))
	}
	return header
}

func ensureCSVHeader(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil // 既存ファイルがあればヘッダー作成はスキップ
	} else if !os.IsNotExist(err) {
		return err
	}

	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()
	return w.Write(csvHeader())
}

// readCompletedIndices はCSVを読み，既に完了済みのTrialインデックス集合を返す。
// 並列実行では完了順がインデックス順と一致しないため，連番カウントではなく
// 実際のTrial列の値で判定する。
func readCompletedIndices(path string) (map[int]bool, error) {
	completed := map[int]bool{}

	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return completed, nil
		}
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	for i, row := range rows {
		if i == 0 || len(row) == 0 {
			continue // header
		}
		idx, err := strconv.Atoi(row[0])
		if err != nil {
			continue
		}
		completed[idx] = true
	}
	return completed, nil
}

// readAllValueRows はCSV全件から，Q1..Q19の数値列のみを抜き出して返す（グラフ生成用）。
func readAllValueRows(path string) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	rawRows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	var rows [][]int
	for i, row := range rawRows {
		if i == 0 {
			continue // header
		}
		if len(row) < 3+NumQuestions {
			continue
		}
		vals := make([]int, NumQuestions)
		ok := true
		for q := 0; q < NumQuestions; q++ {
			v, err := strconv.Atoi(row[3+q])
			if err != nil {
				ok = false
				break
			}
			vals[q] = v
		}
		if ok {
			rows = append(rows, vals)
		}
	}
	return rows, nil
}

// runResultWriter は単一ゴルーチンとしてCSVへの追記を行う（書き込みの競合を避けるため）。
// 各結果が届くごとにファイルを開いて1行追記しFlushする（中断時の安全性を元のPython版に揃える）。
func runResultWriter(csvPath string, results <-chan TrialResult, total int) {
	writtenSoFar := 0
	// 既存の完了件数を起点として表示するため，事前にカウント。
	if existing, err := readCompletedIndices(csvPath); err == nil {
		writtenSoFar = len(existing)
	}

	for res := range results {
		if err := appendCSVRow(csvPath, res); err != nil {
			log.Printf("Trial %d: CSV書き込みに失敗: %v", res.Index, err)
			continue
		}
		writtenSoFar++
		fmt.Printf("Trial %d/%d (%s): %v\n", writtenSoFar, total, res.Persona.Type, res.Values)
	}
}

func appendCSVRow(path string, res TrialResult) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	row := make([]string, 0, 3+NumQuestions)
	row = append(row, strconv.Itoa(res.Index), res.Persona.Type, res.Persona.Desc)
	for _, v := range res.Values {
		row = append(row, strconv.Itoa(v))
	}

	w := csv.NewWriter(f)
	if err := w.Write(row); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}
