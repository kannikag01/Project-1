import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# === Verify that this is the file being executed ===
print("Running from:", __file__)

# === Set your CSV paths ===
train_path = r"C:\image\project\dataset\text\bert_train_combined.csv"
test_path = r"C:\image\project\dataset\text\bert_test_combined.csv"

# === Safety check before reading ===
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test file not found: {test_path}")

# === Load datasets ===
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === Verify column names ===
print("Columns in training CSV:", list(train_df.columns))
if 'combined_text' not in train_df.columns or 'label' not in train_df.columns:
    raise ValueError("CSV must contain 'combined_text' and 'label' columns.")

# === Tokenization ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_batch(df):
    return tokenizer(
        list(df['combined_text']),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

print("Encoding text data...")
train_encodings = encode_batch(train_df)
test_encodings = encode_batch(test_df)

# === Convert labels to tensors ===
train_labels = torch.tensor(train_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

# === Dataset Class ===
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):  # <-- double underscores here
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):  # <-- double underscores here
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):  # <-- double underscores here
        return len(self.labels)

print("Dataset class loaded successfully.")

train_dataset = FakeNewsDataset(train_encodings, train_labels)
test_dataset = FakeNewsDataset(test_encodings, test_labels)

# === Load model ===
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=20,
    save_total_limit=1,
    load_best_model_at_end=True,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("\n=== Training Started (may take time on CPU) ===\n")
trainer.train()

# === Evaluation ===
print("\n=== Evaluating on Test Set ===")
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(-1)
y_true = test_df['label'].values

print("Test Accuracy:", accuracy_score(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=['real', 'fake']))
