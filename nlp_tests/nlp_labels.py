# Exploración inicial: SVM para clasificar cada mensaje a un label específico
# (conceptual_questions, writing_request, provide_context, contextual_questions,
# verification, editing_request, off_topic, misc)

# Se probaron dos metodos (SVM vs Logit) y se eligió el de mejor desempeño.

import json
from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.svm import LinearSVC


TRAIN_PATH = Path("../data/train_13k.jsonl")
TEST_PATH = Path("../data/test_3851.jsonl")

def load_xy(jsonl_path: Path):
    X, y = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = (row.get("prompt") or "").strip()
            full_label = row["llm_label"]["label"]
            main_label = full_label.split(">")[0].strip()
            if prompt:  # saltar vacíos por si acaso
                X.append(prompt)
                y.append(main_label)
    return X, y


def main():
    X_train, y_train = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("Train label distribution:", Counter(y_train))

    # Vectorización simple
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # Clasificador logit
    # clf = LogisticRegression(
    #     max_iter=2000,
    #     n_jobs=None,   # deja None para evitar líos en Windows
    #     class_weight="balanced"
    # )
    # clf.fit(Xtr, y_train)
    # preds = clf.predict(Xte)

    # Clasificador SVM
    clf = LinearSVC(class_weight="balanced")
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification report (macro avg es la que más nos importa):")
    print(classification_report(y_test, preds, digits=4))

    labels_sorted = sorted(set(y_train) | set(y_test))
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    print("Labels order:", labels_sorted)
    print("\nConfusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()