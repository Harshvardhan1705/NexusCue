import csv

path = r".\corpus_csv_files\ISL Corpus sign glosses.csv"

with open(path, encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

for i, row in enumerate(rows, start=1):
    sentence = row["Sentence"]
    gloss = row["SIGN GLOSSES"]
    print(f"{i:03}. {sentence}  ->  {gloss}")