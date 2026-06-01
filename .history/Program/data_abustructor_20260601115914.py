import re
import numpy as np

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
exp_scores = []
psy_scores = []
use_scores = []
none_scores = []

exp_diff = []
psy_diff = []
use_diff = []
none_diff = []

for line in table_text.strip().splitlines():
    cols = [c.strip() for c in line.split("|")[1:-1]]

    my_scores.append(int(cols[1]))

    for cell, score_list, diff_list in zip(
        cols[2:],
        [exp_scores, psy_scores, use_scores, none_scores],
        [exp_diff, psy_diff, use_diff, none_diff]
    ):
        m = re.match(r"(\d+)\s*\(([+-]?\d+)\)", cell.replace("+", ""), cell.replace("-", ""))
        score_list.append(int(m.group(1)))
        diff_list.append(int(m.group(2)))

my_scores = np.array(my_scores)
exp_scores = np.array(exp_scores)
psy_scores = np.array(psy_scores)
use_scores = np.array(use_scores)
none_scores = np.array(none_scores)

exp_diff = np.array(exp_diff)
psy_diff = np.array(psy_diff)
use_diff = np.array(use_diff)
none_diff = np.array(none_diff)

print(my_scores)
print(exp_scores)
print(psy_scores)
print(use_scores)
print(none_scores)
print(exp_diff)
print(psy_diff)
print(use_diff)
print(none_diff)

datasets = {
    "経験付与": exp_scores,
    "心理付与": psy_scores,
    "利用状況": use_scores,
    "付与無し": none_scores,
    "経験差分": exp_diff,
    "心理差分": psy_diff,
    "利用差分": use_diff,
    "付与無し差分": none_diff
}

for name, data in datasets.items():
    print(
        f"{name:10s} "
        f"平均={np.mean(data):.3f} "
        f"標準偏差={np.std(data, ddof=1):.3f}"
    )
