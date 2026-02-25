# Utilidad exploratoria.
# Conteo de líneas en cada dataset para verificar tamaño apropiado.

def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

print("Train lines:", count_lines("../data/train_13k.jsonl"))
print("Test lines:", count_lines("../data/test_3851.jsonl"))