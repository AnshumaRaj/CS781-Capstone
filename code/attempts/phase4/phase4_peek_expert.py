import json

path = r"C:\Users\Anshuman Raj\CS781\phase4data\Phase_4_CJPE\Train\dev.json"
with open(path, encoding="utf-8") as handle:
    rows = json.load(handle)
row = rows[0]
print(sorted(row.keys()))
for key in row:
    if key == "text":
        print("text_words", len(row["text"].split()))
        continue
    value = row[key]
    shown = str(value)[:300].replace("\n", " ")
    print(key, type(value).__name__, shown)
