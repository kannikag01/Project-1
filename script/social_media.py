import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==========================================================
# DATASET PATHS
# ==========================================================
DATASET_PATH_1 = r"C:\image\project\dataset\social\news_articles.csv"
DATASET_PATH_2 = r"C:\image\project\dataset\social\fake_and_real_news.csv"


# ==========================================================
# LOAD CSV DATA
# ==========================================================
def load_csv_data(csv_paths):
    all_dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"Skipping missing CSV: {path}")
            continue

        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]

        # Find text column
        if "text" not in df.columns:
            if "content" in df.columns: df["text"] = df["content"]
            elif "title" in df.columns: df["text"] = df["title"]
            elif "body" in df.columns: df["text"] = df["body"]
            else:
                print(f"No usable text column in {path}, skipping.")
                continue

        # Find label column
        if "label" not in df.columns:
            if "category" in df.columns: df["label"] = df["category"]
            elif "type" in df.columns: df["label"] = df["type"]
            else:
                print(f"No label column in {path}, skipping.")
                continue

        # Clean missing
        df = df.dropna(subset=["text", "label"])

        # Normalize labels
        df["label"] = df["label"].astype(str).str.lower().map(
            {"fake": 0, "false": 0, "real": 1, "true": 1}
        ).fillna(df["label"])

        try:
            df["label"] = df["label"].astype(int)
        except:
            uniq = {v: i for i, v in enumerate(df["label"].unique())}
            df["label"] = df["label"].map(uniq)

        all_dfs.append(df[["text", "label"]])

    if not all_dfs:
        raise ValueError("No valid CSV data found.")

    final_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total samples loaded: {len(final_df)}")
    return final_df


# ==========================================================
# CREATE DATASETS
# ==========================================================
def create_datasets(df):
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    def make_ds(frame):
        texts = frame["text"].astype(str).values
        labels = frame["label"].astype("int32").values
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        return ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return make_ds(train_df), make_ds(val_df), df["text"].astype(str).values


# ==========================================================
# BUILD MODEL WITH TEXTVECTORIZATION
# ==========================================================
def build_text_model(vocab_size=20000, sequence_length=100, embedding_dim=128):

    text_input = tf.keras.Input(shape=(), dtype=tf.string, name="text")

    vectorize_layer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length
    )

    x = vectorize_layer(text_input)
    x = layers.Embedding(vocab_size, embedding_dim)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(text_input, output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model, vectorize_layer


# ==========================================================
# MAIN TRAINING PIPELINE
# ==========================================================
def main():

    df = load_csv_data([DATASET_PATH_1, DATASET_PATH_2])
    train_ds, val_ds, all_texts = create_datasets(df)

    model, vectorizer = build_text_model()

    print("Adapting vectorizer... This may take a moment.")
    vectorizer.adapt(all_texts)

    print("Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=3)

    print("Saving model as social_text_model.keras")
    model.save("social_text_model.keras")

    print("✅ Training complete.")


# ✅ FIXED ENTRY POINT
if __name__ == "__main__":
    main()