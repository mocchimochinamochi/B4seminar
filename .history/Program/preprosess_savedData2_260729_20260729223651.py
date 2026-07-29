import pandas as pd

INPUT_FILE = "Data\\Thesis_nishizawa2025\\Data\\savedData2_ja.csv"
OUTPUT_FILE = "processed_savedData2_ja.csv"

df = pd.read_csv(INPUT_FILE)

# IDがindexに入っている場合だけ戻す
if df.index.name is not None or not str(df.index[0]).isdigit():
    df = df.reset_index(drop=True)

# 念のためID列を確認
print(df[["ID", "言語設定"]].head())

system_col = "システムファイル"

df["人格切り替え"] = df[system_col].astype(str).apply(
    lambda x: "なし" if "ItoI" in x else "あり"
)

df["名前"] = df[system_col].astype(str).apply(
    lambda x: "なし" if "noname" in x else "あり"
)

df["回答者"] = "人間"

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
