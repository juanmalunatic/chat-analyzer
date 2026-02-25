import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("../data/data.jsonl")
TRAIN_OUT  = Path("../data/train_13k.jsonl")
TEST_OUT   = Path("../data/test_3851.jsonl")

TEST_SIZE = 0.22   # Ajustable
RANDOM_STATE = 42

# Lee línea por linea y agrupa los mensajes con el mismo chatId
def load_grouped_by_chat(path: Path):
    """
    Returns:
        dict: chatId -> list of raw JSON lines (as strings)
    """
    grouped = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chat_id = row["chatId"]
            grouped[chat_id].append(line)

    return grouped


def main():

    # Carga el dataset y agrupa por chatId
    print("Loading dataset...")
    grouped = load_grouped_by_chat(INPUT_PATH)

    chat_ids = list(grouped.keys())
    print(f"Total conversations: {len(chat_ids)}")

    # Usa sklearn para segmentar en train/test data sin mezclar chatIds
    print("Splitting by conversation...")
    train_ids, test_ids = train_test_split(
        chat_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(f"Train conversations: {len(train_ids)}")
    print(f"Test conversations: {len(test_ids)}")

    # Escribe datasets finales a archivo
    print("Writing train file...")
    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for chat_id in train_ids:
            for line in grouped[chat_id]:
                f.write(line)

    print("Writing test file...")
    with TEST_OUT.open("w", encoding="utf-8") as f:
        for chat_id in test_ids:
            for line in grouped[chat_id]:
                f.write(line)

    print("Done.")


if __name__ == "__main__":
    main()