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

for line in table_text.strip().splitlines():
    cols = [c.strip() for c in line.split("|")[1:-1]]

    my_scores.append(int(cols[1]))

    for cell, score_list in zip(
        cols[2:],
        [exp_scores, psy_scores, use_scores, none_scores]
    ):
        score = int(re.match(r"(\d+)", cell).group(1))
        score_list.append(score)

my_scores = np.array(my_scores)
exp_scores = np.array(exp_scores)
psy_scores = np.array(psy_scores)
use_scores = np.array(use_scores)
none_scores = np.array(none_scores)

# 差分を再計算
exp_diff = exp_scores - my_scores
psy_diff = psy_scores - my_scores
use_diff = use_scores - my_scores
none_diff = none_scores - my_scores



for name, diff in [
    ("経験付与", exp_diff),
    ("心理付与", psy_diff),
    ("利用状況", use_diff),
    ("付与無し", none_diff)
]:
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))

    print(
        f"{name}: "
        f"RMSE={rmse:.3f}, "
        f"MAE={mae:.3f}, "
        f"平均差={diff.mean():.3f}"
    )
