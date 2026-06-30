package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	plotWidth   = 640
	plotHeight  = 420
	plotMarginL = 60
	plotMarginR = 30
	plotMarginT = 50
	plotMarginB = 60
)

// renderBarChartSVG は 0〜10 の度数分布(counts, len=11)を表す単純な縦棒グラフのSVG文字列を生成する．
// 外部ライブラリに依存せず，テキスト(目盛り・ラベル)も含めて1枚のSVGとして出力する．
func renderBarChartSVG(title string, counts []int) string {
	innerW := plotWidth - plotMarginL - plotMarginR
	innerH := plotHeight - plotMarginT - plotMarginB

	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	if maxCount == 0 {
		maxCount = 1
	}

	n := len(counts)
	barSlot := float64(innerW) / float64(n)
	barWidth := barSlot * 0.7

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" font-family="Helvetica, Arial, sans-serif">`, plotWidth, plotHeight))
	sb.WriteString(fmt.Sprintf(`<rect x="0" y="0" width="%d" height="%d" fill="white"/>`, plotWidth, plotHeight))

	// タイトル
	sb.WriteString(fmt.Sprintf(`<text x="%d" y="24" font-size="16" text-anchor="middle" fill="#222">%s</text>`,
		plotWidth/2, escapeXML(title)))

	// 軸
	axisX0 := plotMarginL
	axisY0 := plotHeight - plotMarginB
	sb.WriteString(fmt.Sprintf(`<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="1.5"/>`,
		axisX0, axisY0, axisX0+innerW, axisY0)) // x軸
	sb.WriteString(fmt.Sprintf(`<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#333" stroke-width="1.5"/>`,
		axisX0, plotMarginT, axisX0, axisY0)) // y軸

	// y軸目盛り（5分割）
	const yTicks = 5
	for i := 0; i <= yTicks; i++ {
		val := maxCount * i / yTicks
		y := axisY0 - int(float64(innerH)*float64(i)/float64(yTicks))
		sb.WriteString(fmt.Sprintf(`<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#ddd" stroke-width="1"/>`,
			axisX0, y, axisX0+innerW, y))
		sb.WriteString(fmt.Sprintf(`<text x="%d" y="%d" font-size="11" text-anchor="end" fill="#555">%d</text>`,
			axisX0-6, y+4, val))
	}

	// バー本体 + x軸ラベル
	for i, c := range counts {
		barH := float64(innerH) * float64(c) / float64(maxCount)
		x := float64(axisX0) + float64(i)*barSlot + (barSlot-barWidth)/2
		y := float64(axisY0) - barH
		sb.WriteString(fmt.Sprintf(`<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#4C78A8"/>`,
			x, y, barWidth, barH))
		if c > 0 {
			sb.WriteString(fmt.Sprintf(`<text x="%.1f" y="%.1f" font-size="10" text-anchor="middle" fill="#222">%d</text>`,
				x+barWidth/2, y-4, c))
		}
		// 目盛りラベル (0〜10)
		sb.WriteString(fmt.Sprintf(`<text x="%.1f" y="%d" font-size="11" text-anchor="middle" fill="#333">%d</text>`,
			x+barWidth/2, axisY0+18, i))
	}

	// 軸ラベル
	sb.WriteString(fmt.Sprintf(`<text x="%d" y="%d" font-size="12" text-anchor="middle" fill="#333">Rating (0-10)</text>`,
		plotWidth/2, plotHeight-15))
	sb.WriteString(fmt.Sprintf(`<text x="14" y="%d" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 14 %d)">Frequency</text>`,
		plotHeight/2, plotHeight/2))

	sb.WriteString(`</svg>`)
	return sb.String()
}

func escapeXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}

// generatePlots は完了済みCSV全件を読み込み，設問ごとの度数分布SVGを plotsDir に書き出す．
func generatePlots(rows [][]int, plotsDir string) error {
	if err := os.MkdirAll(plotsDir, 0o755); err != nil {
		return fmt.Errorf("plotsディレクトリの作成に失敗: %w", err)
	}

	n := len(rows)
	for q := 0; q < NumQuestions; q++ {
		counts := make([]int, 11)
		for _, row := range rows {
			v := row[q]
			if v >= 0 && v <= 10 {
				counts[v]++
			}
		}
		title := fmt.Sprintf("Question %d (N=%d with Personas)", q+1, n)
		svg := renderBarChartSVG(title, counts)
		outPath := filepath.Join(plotsDir, fmt.Sprintf("question_%d.svg", q+1))
		if err := os.WriteFile(outPath, []byte(svg), 0o644); err != nil {
			return fmt.Errorf("question_%d.svg の書き込みに失敗: %w", q+1, err)
		}
	}
	return nil
}
