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
