import time
import os
import re
import csv
import random
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv

# google.genai が使えない環境向けにフォールバックを用意
try:
    from google import genai
    MOCK_GENAI = False
except Exception:
    MOCK_GENAI = True
    class _MockResponse:
        def __init__(self, text):
            self.text = text

    class _MockModels:
        def generate_content(self, model, contents, config):
            vals = [str(random.randint(0, 10)) for _ in range(19)]
            return _MockResponse(", ".join(vals))

    class _MockClient:
        def __init__(self, api_key=None):
            self.models = _MockModels()

    class genai:
        Client = _MockClient

# .envファイルから環境変数を読み込む
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
if not API_KEYS or not API_KEYS[0]:
    print(f"エラー: {env_path} または環境変数に GEMINI_API_KEYS が正しく設定されていません。")
    if MOCK_GENAI:
        print("モックモードにフォールバックします。外部API呼び出しは行われません。")
        API_KEYS = ["MOCK"]
    else:
        print("実環境での実行には GEMINI_API_KEYS が必要です。\n終了します。")
        exit(1)


# =========================================================
# ---- 対話パターンの自動読み込み・タグ付け機構 ----
# =========================================================
# ファイル名の命名規則:
#   chat{SWITCH}_{NAME}_js.(txt|md)
#     SWITCH: "ItoE" = セッション継続（人格切り替えなし）
#             "ItoI" = セッション切替（人格切り替えあり）
#     NAME  : "name"   = LLMが名前を名乗る
#             "noname" = LLMが名前を名乗らない
#
# ※ この対応関係が意図と異なる場合は、下の TAG_RULES を書き換えるだけで
#    判定ロジック全体に反映されます。
DIALOGUE_DIR = os.path.join(os.path.dirname(__file__), "dialogues")

FILENAME_PATTERN = re.compile(
    r"^chat(?P<switch>ItoE|ItoI)_(?P<name>name|noname)_js\.(txt|md)$"
)

TAG_RULES = {
    "switch": {"ItoE": "なし", "ItoI": "あり"},   # 人格切り替え
    "name":   {"name": "あり", "noname": "なし"},  # 名前の有無
}


def load_dialogues(dialogue_dir):
    """dialogues フォルダ内のファイルをスキャンし、ファイル名からタグを自動判定して読み込む"""
    dialogues = []
    if not os.path.isdir(dialogue_dir):
        raise FileNotFoundError(f"対話フォルダが見つかりません: {dialogue_dir}")

    for fname in sorted(os.listdir(dialogue_dir)):
        m = FILENAME_PATTERN.match(fname)
        if not m:
            print(f"警告: 命名規則に一致しないためスキップします: {fname}")
            continue

        fpath = os.path.join(dialogue_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()

        switch_key = m.group("switch")
        name_key = m.group("name")

        dialogues.append({
            "filename": fname,
            "text": text,
            "persona_switch": TAG_RULES["switch"][switch_key],  # "あり" / "なし"
            "has_name": TAG_RULES["name"][name_key],            # "あり" / "なし"
        })

    if not dialogues:
        raise ValueError(f"{dialogue_dir} 内に有効な対話ファイルが見つかりませんでした。")

    return dialogues


DIALOGUES = load_dialogues(DIALOGUE_DIR)
print(f"読み込んだ対話パターン: {len(DIALOGUES)}件")
for d in DIALOGUES:
    print(f"  - {d['filename']} (人格切り替え: {d['persona_switch']}, 名前: {d['has_name']})")


# 質問文の共通テンプレート（対話本文は {dialogue} に差し込む）
QUESTION_TEMPLATE = """
以下のアンケートについて回答してください．思考過程やセクションは不要で，「1,2,3,4,5」のように，半角スペースなしの数字のみ，カンマ区切りで回答してください．

-アンケート-
{dialogue}

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

N_TOTAL = 100

AGE_GROUPS = [
    ("20-29", 1/7), ("30-39", 1/7), ("40-49", 1/7), ("50-59", 1/7),
    ("60-69", 1/7), ("70-79", 1/7), ("80-89", 1/7),
]
DEFAULT_MALE_RATIO = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="Run persona survey with random dialogue selection and demographics")
    p.add_argument("--randomize-demographics", dest="randomize_demographics", action="store_true", default=True)
    p.add_argument("--no-randomize-demographics", dest="randomize_demographics", action="store_false")
    p.add_argument("--invert-demographics", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--male-ratio", type=float, default=DEFAULT_MALE_RATIO)
    return p.parse_args()


args = parse_args()
if args.seed is not None:
    random.seed(args.seed)

trials = []
for p in PERSONAS:
    count = round(N_TOTAL * p["weight"])
    for _ in range(count):
        trials.append(p)

while len(trials) < N_TOTAL:
    trials.append(PERSONAS[0])
while len(trials) > N_TOTAL:
    trials.pop()


def build_demographic_lists(total, age_groups, male_ratio, invert=False, shuffle=False):
    ages = []
    groups = list(age_groups)
    if invert:
        groups = list(reversed(groups))
    for label, w in groups:
        cnt = round(total * w)
        ages.extend([label] * cnt)

    while len(ages) < total:
        ages.append(groups[0][0])
    while len(ages) > total:
        ages.pop()

    male_cnt = round(total * (1 - male_ratio if invert else male_ratio))
    genders = ["Male"] * male_cnt + ["Female"] * (total - male_cnt)

    if shuffle:
        combined = list(zip(ages, genders))
        random.shuffle(combined)
        ages, genders = zip(*combined)
        ages = list(ages)
        genders = list(genders)

    return ages, genders


age_list, gender_list = build_demographic_lists(
    N_TOTAL, AGE_GROUPS, args.male_ratio,
    invert=args.invert_demographics, shuffle=args.randomize_demographics,
)

CSV_FILE = "survey_results_persona.csv"

CSV_HEADER = (
    ["Trial", "PersonaType", "PersonaDesc", "Age", "Gender",
     "DialogueFile", "PersonaSwitch", "HasName"]
    + [f"Q{i+1}" for i in range(19)]
)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

completed_trials = 0
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        completed_trials = sum(1 for line in f) - 1

print(f"Total trials to run: {len(trials)}")
if completed_trials > 0:
    print(f"Resuming from trial {completed_trials}...")

current_key_idx = 0
client = genai.Client(api_key=API_KEYS[current_key_idx])

results = []

for i, persona in enumerate(trials):
    if i < completed_trials:
        continue

    age = age_list[i]
    gender = gender_list[i]

    # ---- 対話パターンをランダムに1つ選択 ----
    dialogue = random.choice(DIALOGUES)
    question_text = QUESTION_TEMPLATE.format(dialogue=dialogue["text"])

    prompt = (
        f"あなたは{persona['desc']}の日本人です。性格タイプは{persona['type']}です。"
        f"年齢は{age}、性別は{gender}です。\n{question_text}"
    )

    max_retries = 3
    attempts = 0
    while attempts < max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config={"temperature": 1.0},
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
                time.sleep(1)
                continue

            results.append(values)

            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [i, persona['type'], persona['desc'], age, gender,
                     dialogue["filename"], dialogue["persona_switch"], dialogue["has_name"]]
                    + values
                )

            print(
                f"Trial {i+1}/{N_TOTAL} ({persona['type']}, {age}, {gender}, "
                f"dialogue={dialogue['filename']}): {values}"
            )
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

# ---- グラフ生成 ----
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        q_start = next((i for i, col in enumerate(header) if col.startswith("Q")), 8)
        results = []
        for row in reader:
            if len(row) <= q_start:
                continue
            try:
                results.append(list(map(int, row[q_start:])))
            except ValueError:
                continue

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
        plt.savefig(f"plots/question_{i+1}.png")
        plt.close()
    print("すべてのグラフを 'plots' フォルダに保存しました。")