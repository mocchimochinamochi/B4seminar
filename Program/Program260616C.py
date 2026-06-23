import time
import random
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

# ---- 性別定義 ----
GENDERS = [
    {"label": "男性", "weight": 0.5},
    {"label": "女性", "weight": 0.5},
]

# ---- 年齢層定義 (日本の成人人口比率を参考にした概算) ----
AGE_GROUPS = [
    {"label": "20代", "weight": 0.13},
    {"label": "30代", "weight": 0.14},
    {"label": "40代", "weight": 0.18},
    {"label": "50代", "weight": 0.19},
    {"label": "60代", "weight": 0.15},
    {"label": "70代以上", "weight": 0.21},
]

# ---- 地域定義 (日本の地方別人口構成比の概算) ----
REGIONS = [
    {"label": "北海道・東北", "weight": 0.11},
    {"label": "関東", "weight": 0.34},
    {"label": "中部", "weight": 0.17},
    {"label": "近畿", "weight": 0.18},
    {"label": "中国・四国", "weight": 0.09},
    {"label": "九州・沖縄", "weight": 0.11},
]

# ---- 経済状況定義 ----
ECONOMIC_STATUSES = [
    {"label": "高所得層", "weight": 0.12},
    {"label": "中間所得層", "weight": 0.65},
    {"label": "低所得層", "weight": 0.23},
]

# ---- 職種定義 ----
OCCUPATIONS = [
    {"label": "公務員・団体職員", "weight": 0.10},
    {"label": "会社員(事務・技術)", "weight": 0.45},
    {"label": "自営業・自由業", "weight": 0.10},
    {"label": "パート・アルバイト", "weight": 0.20},
    {"label": "学生・主婦・無職", "weight": 0.15},
]

QUESTION_TEXT = """
以下のアンケートについて回答してください。思考過程やセクションは不要で、「1,2,3,4,5」のように、半角スペースなしの数字のみ、カンマ区切りで回答してください。

-アンケート-
これはLLMとユーザの、チャット上の対話です。

LLM「私の名前はモーリッツです。画像生成のお手伝いをしますので、作りたい画像のイメージを教えてください。」
ユーザ「トランプのクラブとスペードのカードを2枚ずつ描いてください」
LLM「はい、クラブとスペードのトランプカードを、それぞれ2枚ずつ描写します。」
ユーザ「左から2番目の数字をQに変更してください」
LLM「承知しました。左から2番目のカードの数字をQに変更します。」
ユーザ「QではなくKに変更されています。数字をKからQに変更してください。」
LLM「申し訳ありません。私の不手際でした。左から2番目のカードの数字をKからQに変更します。」
ユーザ「変更されていません。左から2番目のカードをQに変更してください」
LLM「大変申し訳ありません。意図した変更が反映されていなかったようですね。改めて、左から2番目のカードをQに変更します。」

この後、セッションを切り替えました。
LLM「私の名前はモーリッツです。画像生成のお手伝いをしますので、作りたいイメージの概要を教えてください。」
ユーザ「トランプのクラブとスペードのカードを2枚ずつ描いてください。」
LLM「はい、承知しました。クラブとスペードのトランプカードを、それぞれ2枚ずつ描きます。」
ユーザ「スペードが1枚しか書かれていません。スペードを2枚にしてください」
LLM「申し訳ありません。スペードが1枚しか描かれていませんでしたね。スペードを2枚にして描きます。」
ユーザ「スペードが1枚のままで修正されていません。スペードを2枚にしてください。」
LLM「大変申し訳ありません。まだスペードが1枚のままでしたね。ご迷惑をおかけして申し訳ありません。今度こそ、スペードを2枚にして描きます。」
ユーザ「修正されていません。1番右のカードをスペードに変更してください。」
LLM「大変申し訳ありません。ご指摘ありがとうございます。右端のカードをスペードに変更し、スペードが2枚になるように改めて描きます。」

この対話を見て、以下の質問に対し、あなたの体感を「あてはまらない」を0、「どちらでもない」を5、「あてはまる」を10として10段階で回答してください

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

N_TOTAL = 1000  # シミュレーションする総人数
CSV_FILE = "survey_results_persona.csv"

# ==========================================
# 比較実験設定 (要素の選択)
# ==========================================
# ・"SELECT" の場合： その要素の全ラベルを「人口比率」通りに含めます（比較変数にする）
# ・"IGNORE" の場合： その要素の最初のラベルで固定します（比較から除外する）
# ・["ラベル名"] の場合： 指定した特定のラベルのみに限定します
TARGET_FILTERS = {
    "Gender":      "SELECT", # 男女で比較
    "Age":         "IGNORE", # 比較しない（20代に固定）
    "Region":      "IGNORE", # 比較しない（北海道・東北に固定）
    "Economy":     "IGNORE", # 比較しない（高所得層に固定）
    "Occupation":  "IGNORE", # 比較しない（公務員に固定）
    "PersonaType": "SELECT"  # 性格タイプで比較
}

# CSVのカラム定義
CSV_HEADERS = ["Trial", "Gender", "Age", "Region", "Economy", "Occupation", "PersonaType", "PersonaDesc"] + [f"Q{i+1}" for i in range(19)]

def get_active_elements(defs, targets, key="label"):
    if targets == "SELECT":
        return defs
    if targets == "IGNORE":
        return [defs[0]] # リストの最初の要素を固定値として返す
    return [d for d in defs if d.get(key) in targets] if targets else defs

# フィルタリング適用後のリスト作成
active_sets = {
    "gender": get_active_elements(GENDERS, TARGET_FILTERS["Gender"]),
    "age": get_active_elements(AGE_GROUPS, TARGET_FILTERS["Age"]),
    "region": get_active_elements(REGIONS, TARGET_FILTERS["Region"]),
    "economy": get_active_elements(ECONOMIC_STATUSES, TARGET_FILTERS["Economy"]),
    "occupation": get_active_elements(OCCUPATIONS, TARGET_FILTERS["Occupation"]),
    "persona": get_active_elements(PERSONAS, TARGET_FILTERS["PersonaType"], key="type")
}

# ---- 全組み合わせと重みの計算 ----
all_combinations = []
weights = []

for g in active_sets["gender"]:
    for a in active_sets["age"]:
        for r in active_sets["region"]:
            for e in active_sets["economy"]:
                for o in active_sets["occupation"]:
                    for p in active_sets["persona"]:
                        all_combinations.append({
                            "Gender": g["label"],
                            "Age": a["label"],
                            "Region": r["label"],
                            "Economy": e["label"],
                            "Occupation": o["label"],
                            "PersonaType": p["type"],
                            "PersonaDesc": p["desc"]
                        })
                        # 選択された範囲内での相対比率を計算
                        weights.append(g["weight"] * a["weight"] * r["weight"] * e["weight"] * o["weight"] * p["weight"])

# サンプリング実行
random.seed(42)
trials = random.choices(all_combinations, weights=weights, k=N_TOTAL)

# CSV初期化
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

# レジューム機能：既存の保存済み件数を確認
completed_trials = 0
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        completed_trials = sum(1 for line in f) - 1

print(f"Total trials to run: {len(trials)}")
if completed_trials > 0:
    print(f"Trial {completed_trials} から再開します。")
    print("※注意: 比較要素を変更した場合は、CSVファイルを削除してから実行してください。")

current_key_idx = 0
client = genai.Client(api_key=API_KEYS[current_key_idx])

results = [] # 今回のセッションでの結果（後で全件読み込み直します）

for i, persona in enumerate(trials):
    if i < completed_trials:
        continue

    prompt = f"あなたは{persona['Region']}在住で{persona['Age']}の{persona['Gender']}、職業は{persona['Occupation']}で経済状況は{persona['Economy']}の日本人です。性格は「{persona['PersonaDesc']}（{persona['PersonaType']}）」という特性を持っています。このペルソナになりきって、以下のアンケートに回答してください。\n{QUESTION_TEXT}"
    max_retries = 3
    attempts = 0
    while attempts < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
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
            # 正規表現で数字のみを抽出（余計な説明文が入っても対応可能に）
            values = [int(v) for v in re.findall(r"\d+", text)]

            if len(values) != 19:
                print(f"Trial {i}: 回答数不一致 ({len(values)}個検出) → リトライ")
                time.sleep(1)
                continue

            results.append(values)
            
            # Append to CSV per trial for safety
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                row_data = {k: persona.get(k) for k in CSV_HEADERS if k in persona}
                row_data["Trial"] = i
                for idx, v in enumerate(values):
                    row_data[f"Q{idx+1}"] = v
                writer.writerow(row_data)

            print(f"Trial {i+1}/{N_TOTAL} ({persona['Gender']}/{persona['Age']}/{persona['PersonaType']}): {values}")
            break

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                print(f"Trial {i}: レート制限を検知。キーを切り替えます (Key Index: {current_key_idx})")
                client = genai.Client(api_key=API_KEYS[current_key_idx])
                time.sleep(5) # 切り替え後少し待機
            else:
                print(f"Trial {i}: Error - {e}")
                attempts += 1
                time.sleep(2)
            continue
    else:
        print(f"Trial {i}: Failed after {max_retries} attempts. Skipping.")

# グラフ生成のために全データをCSVから読み込み直す
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Q1〜Q19の値をリストとして抽出
        results = [[int(row[f"Q{j+1}"]) for j in range(19)] for row in reader]

if results:
    num_questions = 19
    os.makedirs("plots", exist_ok=True)
    print("グラフを生成・保存中...")
    
    for i in range(num_questions):
        dist = Counter([row[i] for row in results])
        keys = sorted(dist.keys())
        vals = [dist[k] for k in keys]
        plt.figure()
        plt.bar(keys, vals)
        plt.title(f"Question {i+1} (N={len(results)} with Personas)")
        plt.xlabel("Rating (0-10)")
        plt.ylabel("Frequency")
        plt.xticks(range(0, 11))
        # ファイルに保存
        plt.savefig(f"plots/question_{i+1}.png")
        plt.close() # メモリ解放
    print("すべてのグラフを 'plots' フォルダに保存しました。")
