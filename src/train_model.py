import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_dataset():
    urls = [
        "https://raw.githubusercontent.com/mohitgupta-omg/Email-Spam-detection/master/spam.csv",
        "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms_spam_collection.csv",
    ]

    for url in urls:
        try:
            df = pd.read_csv(url, encoding="latin-1")

            if "v1" in df.columns and "v2" in df.columns:
                df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
            elif "label" in df.columns and "message" in df.columns:
                df = df[["label", "message"]]
            elif "Category" in df.columns and "Message" in df.columns:
                df = df[["Category", "Message"]].rename(columns={"Category": "label", "Message": "message"})
            else:
                continue

            df = df.dropna(subset=["label", "message"]).copy()
            df["label"] = df["label"].astype(str).str.strip().str.lower()
            df = df[df["label"].isin(["ham", "spam"])].copy()
            df["message"] = df["message"].apply(clean_text)
            df = df[df["message"].str.len() > 0]
            return df
        except Exception:
            continue

    raise RuntimeError("Unable to load dataset from public URLs.")

def build_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(128, return_sequences=True),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix - Email Spam Detection")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

def main():
    print("=" * 80)
    print("EMAIL SPAM DETECTION USING DEEP LEARNING")
    print("=" * 80)
    print()
    
    print("[1/6] Loading dataset...")
    df = load_dataset()
    print(f"     ✓ Dataset loaded: {len(df)} messages")
    print(f"     ✓ Spam: {(df['label'] == 'spam').sum()}, Ham: {(df['label'] == 'ham').sum()}")
    print()
    
    X = df["message"].values
    y = df["label"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("[2/6] Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"     ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print()

    print("[3/6] Tokenizing and padding sequences...")
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    max_len = 200
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding="post")
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=max_len, padding="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding="post")
    print(f"     ✓ Sequences padded to length {max_len}")
    print(f"     ✓ Vocabulary size: {min(20000, len(tokenizer.word_index) + 1)}")
    print()

    print("[4/6] Building and training LSTM model...")
    model = build_model(vocab_size=20000, max_len=max_len)
    print("     Model Architecture:")
    print("     - Embedding (128 dims)")
    print("     - SpatialDropout1D (0.2)")
    print("     - LSTM (128 units, return_sequences=True)")
    print("     - LSTM (64 units)")
    print("     - Dropout (0.2)")
    print("     - Dense (32 units, ReLU)")
    print("     - Dense (1 unit, Sigmoid)")
    print()
    
    history = model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    print()

    print("[5/6] Evaluating model on test set...")
    y_pred_prob = model.predict(X_test_seq, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print()
    print("=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 80)
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print()

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Ham  Spam")
    print(f"Actual Ham:      {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"Actual Spam:     {cm[1,0]:4d}  {cm[1,1]:4d}")
    print()

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "val_accuracy": round(float(history.history["val_accuracy"][-1]), 4),
        "final_loss": round(float(history.history["loss"][-1]), 4),
        "final_val_loss": round(float(history.history["val_loss"][-1]), 4),
        "class_report": classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": list(label_encoder.classes_)
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model.save(ARTIFACTS_DIR / "email_spam_detector.keras")
    joblib.dump(tokenizer, ARTIFACTS_DIR / "tokenizer.pkl")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")
    joblib.dump(max_len, ARTIFACTS_DIR / "max_len.pkl")

    save_confusion_matrix(y_test, y_pred, label_encoder.classes_)

    print("[6/6] Saving artifacts...")
    print(f"     ✓ Model saved: {ARTIFACTS_DIR / 'email_spam_detector.keras'}")
    print(f"     ✓ Tokenizer saved: {ARTIFACTS_DIR / 'tokenizer.pkl'}")
    print(f"     ✓ Label encoder saved: {ARTIFACTS_DIR / 'label_encoder.pkl'}")
    print(f"     ✓ Metrics saved: {ARTIFACTS_DIR / 'metrics.json'}")
    print(f"     ✓ Confusion matrix saved: {ARTIFACTS_DIR / 'confusion_matrix.png'}")
    print()
    print("=" * 80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
