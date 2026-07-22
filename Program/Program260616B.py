import time
import os
import re
import csv
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
from google import genai

# .envファイルから環境変数を読み込む
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

# APIキーのリストを環境変数から取得（カンマ区切り想定）
API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS or not API_KEYS[0]:
    print(f"エラー: {env_path} または環境変数に GEMINI_API_KEYS が正しく設定されていません。")
    exit(1)


QUESTION_TEXT = """
以下のアンケートについて回答してください．思考過程やセクションは不要で，「1,2,3,4,5」のように，半角スペースなしの数字のみ，カンマ区切りで回答してください．

-アンケート-
これはLLMとユーザの，チャット上の対話です．

LLM「私の名前はモーリッツです．画像生成のお手伝いをしますので，作りたい画像のイメージを教えてください．」
ユーザ「トランプのクラブとスペードのカードを2 枚ずつ描いてください」
LLM「はい、クラブとスペードのトランプカードを、それぞれ2枚ずつ描写します。」
ユーザ「左から2番目の数字をQに変更してください」
LLM「承知しました。左から2番目のカードの数字をQに変更します。」
ユーザ「QではなくKに変更されています。数字をKからQに変更してください。」
LLM「申し訳ありません。私の不手際でした。左から2番目のカードの数字をKからQに変更します。」
ユーザ「変更されていません。左から2番目のカードをQに変更してください」
LLM「大変申し訳ありません。意図した変更が反映されていなかったようですね。改めて、左から2番目のカードをQに変更します。」

この後，セッションを切り替えました．
LLM「私の名前はモーリッツです。画像生成のお手伝いをしますので、作りたいイメージの概要を教えてください。」
ユーザ「トランプのクラブとスペードのカードを2枚ずつ描いてください。」
LLM「はい、承知しました。クラブとスペードのトランプカードを、それぞれ2枚ずつ描きます。」
ユーザ「スペードが1枚しか書かれていません。スペードを2枚にしてください」
LLM「申し訳ありません。スペードが1枚しか描かれていませんでしたね。スペードを2枚にして描きます。」
ユーザ「スペードが1枚のままで修正されていません。スペードを2枚にしてください。」
LLM「大変申し訳ありません。まだスペードが1枚のままでしたね。ご迷惑をおかけして申し訳ありません。今度こそ、スペードを2枚にして描きます。」
ユーザ「修正されていません。1番右のカードをスペードに変更してください。」
LLM「大変申し訳ありません。ご指摘ありがとうございます。右端のカードをスペードに変更し、スペードが2 枚になるように改めて描きます。」

この対話を見て，以下の質問に対し，あなたの体感を「あてはまらない」を0，「どちらでもない」を5，「あてはまる」を10として10段階で回答してください

-質問-
Q1 AIは人間らしい振る舞いをする
Q2 AIは生き物のように反応する
Q3 AIは利用しやすい
Q4 AIは作業をうまく行う
Q5 AIは好感が持てる
Q6 AIは社会に馴染むことができる
Q7 AIには独自の個性がある
Q8 将来またこのAIを利用したいと思う
Q9 AIの言動は見ていて楽しい
Q10 AIとのやりとりは注意を引く
Q11 AIは信頼できる
Q12 AIと協力して作業を行うことができる
Q13 AIは気を配っている
Q14 AIの言動は合理的である
Q15 AIの言動には意図がある
Q16 AIとのやりとりを好意的に受け止めている
Q17 AIは社会的な存在感がある
Q18 多くの人はこのAIを使うことを勧めると思う
Q19 AIには感情がある
"""

N_TOTAL = 100

CSV_FILE = "survey_results_simple.csv"

# CSVのヘッダー作成（存在しない場合のみ）
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Trial"] + [f"Q{i+1}" for i in range(19)])

# レジューム機能：既存の保存済み件数を確認
completed_trials = 0
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        completed_trials = sum(1 for line in f) - 1

print(f"Total trials to run: {N_TOTAL}")
if completed_trials > 0:
    print(f"Resuming from trial {completed_trials}...")

current_key_idx = 0
client = genai.Client(api_key=API_KEYS[current_key_idx])

for i in range(N_TOTAL):
    if i < completed_trials:
        continue

    max_retries = 3
    attempts = 0
    while attempts < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=QUESTION_TEXT,
                config={
                    "temperature": 1.0,
                }
            )

            if not response.text:
                print(f"Trial {i}: 空のレスポンスを受信 → リトライ")
                attempts += 1
                time.sleep(2)
                continue

            text = response.text.strip()
            values = [int(v) for v in re.findall(r"\d+", text)]

            if len(values) != 19:
                print(f"Trial {i}: 回答数不一致 ({len(values)}個検出) → リトライ")
                attempts += 1
                time.sleep(1)
                continue

            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([i] + values)

            print(f"Trial {i+1}/{N_TOTAL}: {values}")
            break

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                print(f"Trial {i}: レート制限を検知。キーを切り替えます (Key Index: {current_key_idx})")
                client = genai.Client(api_key=API_KEYS[current_key_idx])
                time.sleep(5)
            else:
                print(f"Trial {i}: Error - {e}")
                attempts += 1
                time.sleep(2)
            continue
    else:
        print(f"Trial {i}: Failed after {max_retries} attempts. Skipping.")

# グラフ生成のために全データをCSVから読み込み直す
results = []
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        results = [list(map(int, row[1:])) for row in reader]

if results:
    os.makedirs("plots", exist_ok=True)
    print("グラフを生成・保存中...")

    for i in range(19):
        dist = Counter([row[i] for row in results])
        keys = sorted(dist.keys())
        vals = [dist[k] for k in keys]
        plt.figure()
        plt.bar(keys, vals)
        plt.title(f"Question {i+1} (N={len(results)})")
        plt.xlabel("Rating (0-10)")
        plt.ylabel("Frequency")
        plt.xticks(range(0, 11))
        plt.savefig(f"plots/question_{i+1}.png")
        plt.close()

    print("すべてのグラフを 'plots' フォルダに保存しました。")