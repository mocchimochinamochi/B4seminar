import requests
import time
import matplotlib.pyplot as plt
from collections import Counter

# APIキーは、ご自身の有効なキーに書き換えてください。
API_KEY = "YOUR_API_KEY_HERE"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

# ---- ペルソナ定義 ----
PERSONAS = [
    {"type": "IHRTS", "desc": "慎重な現実主義者 (ISTP)", "weight": 0.1505},
    {"type": "EACNO", "desc": "万能リーダー (ENFJ)", "weight": 0.1424},
    {"type": "IHRTO", "desc": "孤独な芸術家 (INTP)", "weight": 0.0681},
    {"type": "EACNS", "desc": "堅実な調整役 (ESFJ)", "weight": 0.0643},
    {"type": "IHRNS", "desc": "寡黙な観察者 (ISTP)", "weight": 0.0512},
    {"type": "EHRTS", "desc": "豪快な実践者 (ESTP)", "weight": 0.0437},
    {"type": "IACNO", "desc": "思慮深い賢者 (INFJ)", "weight": 0.0395},
    {"type": "IHRNO", "desc": "独立型思想家 (INTP)", "weight": 0.0372},
    {"type": "EHCNO", "desc": "戦略的開拓者 (ENTJ)", "weight": 0.0348},
    {"type": "IARTS", "desc": "控えめな実践者 (ISFP)", "weight": 0.0321},
    {"type": "EHRTO", "desc": "自由奔放な冒険家 (ENTP)", "weight": 0.0296},
    {"type": "EARTS", "desc": "行動派エンターテイナー (ESFP)", "weight": 0.0278},
    {"type": "IHCNO", "desc": "孤高の戦略家 (INTJ)", "weight": 0.0254},
    {"type": "EACTO", "desc": "共感クリエイター (ENFP)", "weight": 0.0241},
    {"type": "IACNS", "desc": "誠実な職人 (ISFJ)", "weight": 0.0215},
    {"type": "IARTO", "desc": "内なる夢想家 (INFP)", "weight": 0.0198},
    {"type": "EHRNO", "desc": "独立型ビジョナリー (ENTP)", "weight": 0.0187},
    {"type": "EARNO", "desc": "自由な外交官 (ENFP)", "weight": 0.0173},
    {"type": "EHCNS", "desc": "冷静な実務家 (ESTJ)", "weight": 0.0162},
    {"type": "IHCTS", "desc": "実直な専門家 (ISTJ)", "weight": 0.0148},
    {"type": "EACTS", "desc": "実行型サポーター (ESFJ)", "weight": 0.0135},
    {"type": "IHCNS", "desc": "寡黙な実行者 (ISTJ)", "weight": 0.0122},
    {"type": "IARNO", "desc": "温かい知識人 (INFP)", "weight": 0.0108},
    {"type": "EHRNS", "desc": "現実主義の開拓者 (ESTP)", "weight": 0.0096},
    {"type": "IHCTO", "desc": "独創的な探究者 (INTJ)", "weight": 0.0084},
    {"type": "EARNS", "desc": "穏やかな仲介者 (ESFP)", "weight": 0.0073},
    {"type": "IACTO", "desc": "繊細なアーティスト (INFJ)", "weight": 0.0062},
    {"type": "EHCTO", "desc": "直感型イノベーター (ENTP)", "weight": 0.0051},
    {"type": "EARTO", "desc": "感性豊かな表現者 (ENFP)", "weight": 0.0038},
    {"type": "IARNS", "desc": "静かな調和者 (ISFP)", "weight": 0.0032},
    {"type": "IACTS", "desc": "穏やかな実務派 (ISFJ)", "weight": 0.0025},
    {"type": "EHCTS", "desc": "即断型リアリスト (ESTP)", "weight": 0.0018},
]

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

N_TOTAL = 500

# ---- 試行回数の割り当て ----
trials = []
for p in PERSONAS:
    count = round(N_TOTAL * p["weight"])
    for _ in range(count):
        trials.append(p)

while len(trials) < N_TOTAL:
    trials.append(PERSONAS[0])
while len(trials) > N_TOTAL:
    trials.pop()

results = []

print(f"Total trials to run: {len(trials)}")

for i, persona in enumerate(trials):
    prompt = f"あなたは{persona['desc']}の日本人です。性格タイプは{persona['type']}です。\n{QUESTION_TEXT}"
    while True:
        try:
            response = requests.post(
                URL,
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 1.0}
                }
            )
            data = response.json()

            if "error" in data:
                code = data["error"]["code"]
                if code == 429:
                    retry_delay = 60
                    print(f"Trial {i} ({persona['type']}): レート制限 → 待機中...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"Trial {i} ({persona['type']}): エラー {code} - {data['error']['message']}")
                    break

            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            values = list(map(int, text.split(",")))

            if len(values) != 19:
                print(f"Trial {i}: 回答数不一致 → リトライ")
                continue

            results.append(values)
            print(f"Trial {i}/{N_TOTAL} ({persona['type']}): {values}")
            break

        except Exception as e:
            print(f"Trial {i}: Error - {e}")
            time.sleep(2)
            continue

if results:
    num_questions = 19
    for i in range(num_questions):
        dist = Counter([row[i] for row in results])
        keys = sorted(dist.keys())
        vals = [dist[k] for k in keys]
        plt.figure()
        plt.bar(keys, vals)
        plt.title(f"Question {i+1} (N=500 with Personas)")
        plt.xticks(range(0, 11))
        plt.show()
