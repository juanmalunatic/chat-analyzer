import csv
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from conversation_features import (
    load_prompts_labels_meta,
    run_sentiment_scoring_transformer,
)

TEST_PATH = Path("../data/test_3851.jsonl")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


def to_ternary_label(d: dict) -> str:
    lab = str(d.get("label", "")).lower()

    if "neg" in lab or "negative" in lab:
        return "NEGATIVE"
    if "pos" in lab or "positive" in lab:
        return "POSITIVE"
    if "neu" in lab or "neutral" in lab:
        return "NEUTRAL"

    if lab == "label_0":
        return "NEGATIVE"
    if lab == "label_2":
        return "POSITIVE"
    return "NEUTRAL"


def run_with_timing(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, dt


def fmt_speed(n, seconds):
    if seconds <= 0:
        return "n/a", "n/a", "n/a"
    prompts_per_sec = n / seconds
    sec_per_1000 = seconds / (n / 1000.0)
    return f"{seconds:.2f}s", f"{sec_per_1000:.2f}s/1k", f"{prompts_per_sec:.2f} prompts/s"


def main():
    X_test, _, _ = load_prompts_labels_meta(TEST_PATH)
    n = len(X_test)
    print(f"Loaded test prompts: {n}")

    timings = {}

    # 1) Proxy truth: RoBERTa
    print("\nRunning RoBERTa (proxy truth)...")
    roberta_out, t = run_with_timing(run_sentiment_scoring_transformer, X_test, model="roberta")
    timings["RoBERTa"] = t
    y_ref = [to_ternary_label(d) for d in roberta_out]

    # 2) BERTweet
    print("\nRunning BERTweet...")
    bertweet_out, t = run_with_timing(run_sentiment_scoring_transformer, X_test, model="bertweet")
    timings["BERTweet"] = t
    y_bertweet = [to_ternary_label(d) for d in bertweet_out]

    # 3) Multilingual
    print("\nRunning Multilingual...")
    multi_out, t = run_with_timing(run_sentiment_scoring_transformer, X_test, model="multilingual")
    timings["Multilingual"] = t
    y_multi = [to_ternary_label(d) for d in multi_out]

    # 4) VADER
    print("\nRunning VADER...")
    vader_out, t = run_with_timing(run_sentiment_scoring_transformer, X_test, model="vader")
    timings["VADER"] = t
    y_vader = [to_ternary_label(d) for d in vader_out]

    # Agreement vs RoBERTa
    print("\n=== Agreement vs RoBERTa (proxy) ===")
    print("BERTweet     acc:", accuracy_score(y_ref, y_bertweet))
    print("Multilingual acc:", accuracy_score(y_ref, y_multi))
    print("VADER        acc:", accuracy_score(y_ref, y_vader))

    print("\nBERTweet report:")
    print(classification_report(y_ref, y_bertweet, digits=4, zero_division=0))
    print("\nMultilingual report:")
    print(classification_report(y_ref, y_multi, digits=4, zero_division=0))
    print("\nVADER report:")
    print(classification_report(y_ref, y_vader, digits=4, zero_division=0))

    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    print("\nLabels order:", labels)

    print("\nConfusion matrix BERTweet vs RoBERTa:")
    print(confusion_matrix(y_ref, y_bertweet, labels=labels))
    print("\nConfusion matrix Multilingual vs RoBERTa:")
    print(confusion_matrix(y_ref, y_multi, labels=labels))
    print("\nConfusion matrix VADER vs RoBERTa:")
    print(confusion_matrix(y_ref, y_vader, labels=labels))

    # Timing summary
    print("\n=== Runtime (end-to-end) ===")
    for name in ["RoBERTa", "BERTweet", "Multilingual", "VADER"]:
        secs = timings[name]
        total_s, s_per_1k, pps = fmt_speed(n, secs)
        print(f"{name:12s} | total: {total_s:>10s} | {s_per_1k:>10s} | {pps:>14s}")

    # Export rows
    out_csv = OUT_DIR / "sentiment_benchmark_rows.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "roberta_label", "bertweet_label", "multilingual_label", "vader_label"])
        for i, (a, b, c, d) in enumerate(zip(y_ref, y_bertweet, y_multi, y_vader)):
            w.writerow([i, a, b, c, d])

    # Export timing CSV (para que quede en outputs)
    out_t = OUT_DIR / "sentiment_benchmark_timings.csv"
    with out_t.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "seconds_total", "seconds_per_1000", "prompts_per_second"])
        for name in ["RoBERTa", "BERTweet", "Multilingual", "VADER"]:
            secs = timings[name]
            _, s_per_1k, pps = fmt_speed(n, secs)
            # guardo numéricos también
            sec_per_1k_num = secs / (n / 1000.0) if secs > 0 else ""
            pps_num = n / secs if secs > 0 else ""
            w.writerow([name, round(secs, 6), round(sec_per_1k_num, 6), round(pps_num, 6)])

    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_t}")


if __name__ == "__main__":
    main()