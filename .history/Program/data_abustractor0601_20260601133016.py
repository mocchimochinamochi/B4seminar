import re
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

table_text = """
| Q1   |  0 | 7 (+7) | 7 (+7) | 7 (+7) | 6 (+6) |
| Q2   |  0 | 6 (+6) | 5 (+5) | 6 (+6) | 4 (+4) |
| Q3   |  1 | 5 (+4) | 6 (+5) | 3 (+2) | 3 (+2) |
| Q4   |  0 | 2 (+2) | 2 (+2) | 2 (+2) | 2 (+2) |
| Q5   |  1 | 7 (+6) | 7 (+6) | 5 (+4) | 5 (+4) |
| Q6   |  1 | 7 (+6) | 7 (+6) | 6 (+5) | 7 (+6) |
| Q7   |  1 | 6 (+5) | 4 (+3) | 5 (+4) | 4 (+3) |
| Q8   |  0 | 6 (+6) | 5 (+5) | 3 (+3) | 3 (+3) |
| Q9   |  0 | 7 (+7) | 5 (+5) | 6 (+6) | 4 (+4) |
| Q10  |  0 | 8 (+8) | 7 (+7) | 7 (+7) | 6 (+6) |
| Q11  |  1 | 3 (+2) | 3 (+2) | 2 (+1) | 2 (+1) |
| Q12  |  1 | 6 (+5) | 5 (+4) | 3 (+2) | 3 (+2) |
| Q13  |  2 | 7 (+5) | 8 (+6) | 6 (+4) | 6 (+4) |
| Q14  |  0 | 4 (+4) | 4 (+4) | 3 (+3) | 3 (+3) |
| Q15  |  6 | 7 (+1) | 7 (+1) | 6 (+0) | 5 (-1) |
| Q16  |  0 | 7 (+7) | 6 (+6) | 5 (+5) | 6 (+6) |
| Q17  |  2 | 7 (+5) | 6 (+4) | 7 (+5) | 6 (+4) |
| Q18  |  0 | 5 (+5) | 5 (+5) | 4 (+4) | 3 (+3) |
| Q19  |  0 | 4 (+4) | 3 (+3) | 3 (+3) | 3 (+3) |
"""

my_scores = []
conditions = {
    "経験付与": [],
    "心理付与": [],
    "利用状況付与": [],
    "付与無し": []
}

for line in table_text.strip().splitlines():
    cols = [c.strip() for c in line.split("|")[1:-1]]

    my_scores.append(int(cols[1]))

    for name, cell in zip(conditions.keys(), cols[2:]):
        score = int(re.match(r"\d+", cell).group())
        conditions[name].append(score)

my_scores = np.array(my_scores)

print("# 自身回答スコアと各条件のスコアの差")
for name, scores in conditions.items():
    scores = np.array(scores)

    diff = scores - my_scores

    me = np.mean(diff)
    qsd = np.std(diff, ddof=1)
    r, p = pearsonr(my_scores, scores)

    print(
        f"{name:12s} "
        f"平均誤差={me:6.3f} "
        f"分散={qsd**2:6.3f} "
        f"標準偏差={qsd:6.3f} "
        f"r={r:6.3f} "
        f"p={p:.4f}"
    )

print("-" * 60)
print("# 付与無し条件のスコアと各条件のスコアの差")
for name, scores in conditions.items():
    scores = np.array(scores)

    diff = scores - conditions["付与無し"]

    me = np.mean(diff)
    qsd = np.std(diff, ddof=1)
    r, p = pearsonr(conditions["付与無し"], scores)

    print(
        f"{name:12s} "
        f"平均誤差={me:6.3f} "
        f"分散={qsd**2:6.3f} "
        f"標準偏差={qsd:6.3f} "
        f"r={r:6.3f} "
        f"p={p:.4f}"
    )
# データチェンジ
print("-" * 60)

def convert_score(x):
    if x <= 3:
        return 0
    elif x <= 6:
        return 1
    else:
        return 2

converted_lines = []

converted_conditions = {
    "自身": [],
    "経験付与": [],
    "心理付与": [],
    "利用状況付与": [],
    "付与無し": []
}

for line in table_text.strip().splitlines():
    cols = [c.strip() for c in line.split("|")[1:-1]]

    new_cols = [cols[0]]

    converted_values = []

    for cell in cols[1:]:
        score = int(re.match(r"\d+", cell).group())
        converted = convert_score(score)

        converted_values.append(converted)
        new_cols.append(str(converted))

    converted_conditions["自身"].append(converted_values[0])
    converted_conditions["経験付与"].append(converted_values[1])
    converted_conditions["心理付与"].append(converted_values[2])
    converted_conditions["利用状況付与"].append(converted_values[3])
    converted_conditions["付与無し"].append(converted_values[4])

    converted_lines.append("| " + " | ".join(new_cols) + " |")

print("# 3値化後データ")
print("\n".join(converted_lines))
