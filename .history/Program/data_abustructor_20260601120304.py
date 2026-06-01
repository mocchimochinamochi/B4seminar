import re
import numpy as np

table_text = """

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
