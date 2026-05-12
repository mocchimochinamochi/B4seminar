import requests
import time
import matplotlib.pyplot as plt
from collections import Counter

# APIキーは情報漏洩に気を付けてください．
API_KEY = "AIzaSyADImZQ3DHqB-DER_s-MWmSQXg5JvNwte4"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-robotics-er-1.5-preview:generateContent?key={API_KEY}"

# 西澤さんの研究をプロンプトにしてみました．低スペのLLMだと正確に回答してくれません
PROMPT = """
あなたは日本国民のある一人です．以下のアンケートについて回答してください．思考過程やセクションは不要で，「1,2,3,4,5」のように，半角スペースなしの数字のみ，カンマ区切りで回答してください．

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

# 繰り返し回数
N_TRIALS = 100

# ---- LLMから回答をもらう ----
results = []

for i in range(N_TRIALS):
    while True:  # 成功するまでリトライ
        try:
            response = requests.post(
                URL,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": PROMPT}]}],
                    "generationConfig": {"temperature": 1.0}
                }
            )
            data = response.json()

            # エラーレスポンスの処理
            if "error" in data:
                code = data["error"]["code"]
                if code == 429:
                    # retryDelayを取得して待機
                    retry_delay = 60
                    for detail in data["error"].get("details", []):
                        if "retryDelay" in detail:
                            retry_delay = int(detail["retryDelay"].replace("s", "")) + 5
                            break
                    print(f"Trial {i}: レート制限 → {retry_delay}秒待機してリトライ")
                    time.sleep(retry_delay)
                    continue
                elif code == 503:
                    print(f"Trial {i}: サーバー混雑 → 10秒待機してリトライ")
                    time.sleep(10)
                    continue
                else:
                    print(f"Trial {i}: 不明なエラー {code} → スキップ")
                    break

            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            values = list(map(int, text.split(",")))

            if len(values) != 19:
                print(f"Trial {i}: 回答数が不正 ({len(values)}個) → リトライ, テキスト: {text}")
                time.sleep(5)
                continue

            results.append(values)
            print(f"Trial {i}: {values}")
            break  # 成功

        except (KeyError, ValueError) as e:
            print(f"Trial {i}: パースエラー - {e} → リトライ")
            time.sleep(5)

if not results:
    print("有効な結果がありません")
    exit()

# ---- 集計 ----
num_questions = len(results[0])
distributions = [Counter() for _ in range(num_questions)]

for row in results:
    for j, val in enumerate(row):
        distributions[j][val] += 1

# ---- グラフ描画 ----
for i, dist in enumerate(distributions):
    keys = sorted(dist.keys())
    vals = [dist[k] for k in keys]

    plt.figure()
    plt.bar(keys, vals)
    plt.title(f"Question {i+1}")
    plt.xlabel("Answer")
    plt.ylabel("Frequency")
    plt.xticks(range(0, 11))
    plt.show()

print("すべてのグラフを表示しました")
