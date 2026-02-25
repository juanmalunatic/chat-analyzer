# Utilidad exploratoria.
# Conteo de labels asignados por la llm para ver cuantas categorías hay en el dataset de entrenamiento.

import json
from collections import Counter

file_path = "../data/train_13k.jsonl"

label_counter = Counter()

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        full_label = data["llm_label"]["label"]

        # Nos quedamos solo con la parte antes del ">"
        main_label = full_label.split(">")[0]

        label_counter[main_label] += 1

print("Total unique labels:", len(label_counter))
print("\nTop 15 labels:\n")

for label, count in label_counter.most_common(15):
    print(f"{label}: {count}")