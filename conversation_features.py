
import json
import csv
import joblib
from collections import defaultdict, Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import torch
from transformers import pipeline

import sys
#print("PYTHON EXE:", sys.executable)
#print("PYTHON VER:", sys.version)
#print("TORCH VER:", torch.__version__)
#print("CUDA AVAIL:", torch.cuda.is_available())

TRAIN_PATH = Path("data/train_13k.jsonl")
TEST_PATH  = Path("data/test_3851.jsonl")
MODEL_PATH = Path("outputs/help_classifier.joblib")
FORCE_RETRAIN = False

# Extraer únicamente [prompt, label, [meta]] de cada línea del dataset.
# (meta contiene otros campos que queremos guardar del dataset original)
def load_prompts_labels_meta(jsonl_path: Path):

    # Helper para retirar la segunda parte del label
    # "contextual_questions>Other" -> "contextual_questions"
    def parse_main_label(full_label: str) -> str:
        return full_label.split(">")[0].strip()

    prompts, labels, meta = [], [], []  # meta: (chat_id, interaction_count)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue

            prompts.append(prompt)
            labels.append(parse_main_label(row["llm_label"]["label"]))

            # acá preservamos el id, interactionCount y timestamps
            meta.append((
                row["chatId"],
                row["interactionCount"],
                row.get("timestamp")
            ))

    return prompts, labels, meta


def run_label_classifier(X_train, y_train, X_test, y_test, meta_test):
    # ------------------------------
    # NLP 1: clasificación de labels
    # ------------------------------
    # Si el modelo está cached, se usa para ahorrar tiempo.
    if MODEL_PATH.exists() and not FORCE_RETRAIN:
        print("Loading cached model...")
        vectorizer, clf = joblib.load(MODEL_PATH)
    else:
        print("Training model...")
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
        Xtr = vectorizer.fit_transform(X_train)

        clf = LinearSVC(class_weight="balanced")
        clf.fit(Xtr, y_train)

        joblib.dump((vectorizer, clf), MODEL_PATH)
        print("Model saved.")

    # Independientemente de si se carga o no, se clasifica la data de test.
    Xte = vectorizer.transform(X_test)
    # Esta es la lista de labels predicha
    preds = clf.predict(Xte)

    # Printeamos la accuracy de esta parte
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification report (macro avg es la que más nos importa):")
    print(classification_report(y_test, preds, digits=4))

    labels_sorted = sorted(set(y_train) | set(y_test))
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    print("Labels order:", labels_sorted)
    print("\nConfusion matrix:")
    print(cm)

    return preds


def normalize_sentiment_label(raw_label: str) -> str:
    """
    Normaliza labels de distintos modelos HF + VADER a:
    POSITIVE / NEUTRAL / NEGATIVE
    """
    lab = (raw_label or "").strip().lower()

    # Transformers comunes
    if "neg" in lab or lab == "label_0":
        return "NEGATIVE"
    if "neu" in lab or lab == "label_1":
        return "NEUTRAL"
    if "pos" in lab or lab == "label_2":
        return "POSITIVE"

    # Fallback razonable
    return "NEUTRAL"


# Buena relación accuracy/performance con multilingual.
# Usar RoBERTa o BERTweet para mejor calidad (más lento).

def run_sentiment_labeling(X_test, model="multilingual"):
    """
    Devuelve: list[str] con labels normalizados
    """
    device = 0 if torch.cuda.is_available() else -1

    if model == "bertweet":
        clf = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            truncation=True,
            padding=True,
            max_length=128,
            device=device
        )
    elif model == "multilingual":
        clf = pipeline(
            "sentiment-analysis",
            model="tabularisai/multilingual-sentiment-analysis",
            truncation=True,
            padding=True,
            max_length=128,
            device=device
        )
    elif model == "roberta":
        clf = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            padding=True,
            max_length=128,
            device=device
        )
    else:
        # VADER: solo label (sin score)
        analyzer = SentimentIntensityAnalyzer()
        out_labels = []
        for t in X_test:
            c = analyzer.polarity_scores(t)["compound"]  # [-1, 1]
            if c <= -0.05:
                out_labels.append("NEGATIVE")
            elif c >= 0.05:
                out_labels.append("POSITIVE")
            else:
                out_labels.append("NEUTRAL")
        return out_labels

    print("pipeline device param:", device)
    print("torch cuda available:", torch.cuda.is_available())
    print("model device:", next(clf.model.parameters()).device)

    batch_size = 128
    out_labels = []
    for i in tqdm(range(0, len(X_test), batch_size), desc="Sentiment (batches)"):
        batch = X_test[i:i + batch_size]
        preds = clf(batch, batch_size=batch_size)
        out_labels.extend([normalize_sentiment_label(p.get("label")) for p in preds])

    return out_labels


def export_turn_level_csv(X_test, y_test, pred_labels, meta_test, sent_labels):
    OUTPUT_TURN = Path("outputs/turn_level_predictions.csv")

    with OUTPUT_TURN.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chatId",
            "interactionCount",
            "timestamp",
            "prompt",
            "true_label",
            "pred_label",
            "sentiment_label",
        ])

        for prompt, true_label, pred_label, (chat_id, interaction_count, ts), s_lab in zip(
            X_test, y_test, pred_labels, meta_test, sent_labels
        ):
            writer.writerow([
                chat_id, interaction_count, ts, prompt, true_label, pred_label, s_lab
            ])

def lexical_ratio(text: str) -> float:
    words = text.split()
    # set remueve duplicados, #unicas/#total
    return (len(set(words)) / len(words)) if words else 0.0


def aggregate_conversations(prompts, preds, meta, sentiment_labels):
    """
    Returns dict: chatId -> list of (prompt, pred_label, interaction_count, ts, sentiment_label)
    Ordenado por interaction_count dentro de cada chat.
    """
    convo = defaultdict(list)

    for prompt, pred_label, (chat_id, interaction_count, ts), sent_lab in zip(
        prompts, preds, meta, sentiment_labels
    ):
        convo[chat_id].append((prompt, pred_label, interaction_count, ts, sent_lab))

    for chat_id in convo:
        convo[chat_id].sort(key=lambda x: x[2])

    return convo

def conversation_metrics(items):
    """
    items: list of (prompt, pred_label, interaction_count)
    """
    n_turns = len(items)

    word_counts = []
    question_flags = []
    unique_ratios = []
    labels = []
    timestamps = []
    sentiment_labels = []

    for prompt, pred_label, interaction_count, ts, sent_lab in items:
        words = prompt.split()
        # contar palabras
        word_counts.append(len(words))

        # contar cuantas veces aparece "?"
        question_flags.append("?" in prompt)

        # contar proporción de palabras únicas
        unique_ratios.append(lexical_ratio(prompt))

        # agregar labels predichos
        labels.append(pred_label)

        # agregar sentimientos
        sentiment_labels.append(sent_lab)

        if ts is not None:
            timestamps.append(ts)

    label_switch_count = sum(
        1 for i in range(1, len(labels)) if labels[i] != labels[i - 1]
    )
    unique_help_types = len(set(labels))
    total_words = sum(word_counts)

    if timestamps:
        # timestamps están en milisegundos → convertir a segundos
        duration_minutes = (max(timestamps) - min(timestamps)) / 1000.0 / 60.0
        duration_minutes = round(duration_minutes, 2)
    else:
        duration_minutes = ""

    dominant_label = Counter(labels).most_common(1)[0][0]
    dominant_sentiment = Counter(sentiment_labels).most_common(1)[0][0] if sentiment_labels else "NEUTRAL"

    return {
        "n_turns": n_turns,
        "duration_minutes": duration_minutes,
        "avg_prompt_len": sum(word_counts) / n_turns,
        "question_ratio": sum(question_flags) / n_turns,
        "avg_unique_ratio": sum(unique_ratios) / n_turns,
        "dominant_help_type": dominant_label,
        "dominant_sentiment": dominant_sentiment,
        "total_words": total_words,
        "label_switch_count": label_switch_count,
        "unique_help_types": unique_help_types,
    }



def main():
    # Crear directorio de output
    Path("outputs").mkdir(exist_ok=True)

    # Se cargan las variables del disco a ser usadas por NLP
    X_train, y_train, _         = load_prompts_labels_meta(TRAIN_PATH)
    X_test , y_test , meta_test = load_prompts_labels_meta(TEST_PATH)

    # NLP1: Clasificación por label
    pred_labels = run_label_classifier(X_train, y_train, X_test, y_test, meta_test)

    # NLP2: Sentiment scores
    print("Sentiment labels ....")
    sent_labels = run_sentiment_labeling(X_test, model="multilingual")

    # Exportar data a nivel de "turnos" (prompts individuales)
    print("Exporting turn-level CSV ....")
    export_turn_level_csv(X_test, y_test, pred_labels, meta_test, sent_labels)

    print("Aggregating data ....")
    # Métricas de agregación: frecuencias y conteo de features NLP
    convo_data = aggregate_conversations(X_test, pred_labels, meta_test, sent_labels)

    # Escritura a disco
    OUTPUT_CONVO = Path("outputs/conversation_level_features.csv")

    with OUTPUT_CONVO.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "chatId",
            "n_turns",
            "duration_minutes",
            "total_words",
            "avg_prompt_len",
            "question_ratio",
            "avg_unique_ratio",
            "label_switch_count",
            "unique_help_types",
            "dominant_help_type",
            "dominant_sentiment",
        ])

        for chat_id, items in convo_data.items():
            m = conversation_metrics(items)
            writer.writerow([
                chat_id,
                m["n_turns"],
                m["duration_minutes"],
                m["total_words"],
                m["avg_prompt_len"],
                m["question_ratio"],
                m["avg_unique_ratio"],
                m["label_switch_count"],
                m["unique_help_types"],
                m["dominant_help_type"],
                m["dominant_sentiment"],
            ])

    print("Total conversations in test:", len(convo_data))
    print("\nSample conversation-level metrics:\n")

    for chat_id, items in list(convo_data.items())[:5]:
        m = conversation_metrics(items)

        print(f"chatId: {chat_id}")
        print(f"  n_turns: {m['n_turns']}")
        print(f"  duration_minutes: {m['duration_minutes']}")
        print(f"  avg_prompt_len: {m['avg_prompt_len']:.2f}")
        print(f"  question_ratio: {m['question_ratio']:.2f}")
        print(f"  avg_unique_ratio: {m['avg_unique_ratio']:.2f}")
        print(f"  dominant_help_type: {m['dominant_help_type']}")
        print(f"  dominant_sentiment: {m['dominant_sentiment']}")
        print("-" * 40)


if __name__ == "__main__":
    main()