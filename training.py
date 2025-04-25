import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from multitask import MultiTaskModel

# Dummy Dataset
class DummyMultiTaskDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels_a, labels_b):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.labels_a = labels_a
        self.labels_b = labels_b

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.sentences[idx], padding="max_length", max_length=32, truncation=True, return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_a": torch.tensor(self.labels_a[idx]),
            "label_b": torch.tensor(self.labels_b[idx]),
            "text": self.sentences[idx]
        }

# Setup
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentences = [
    "The stock market closed higher today.",         # News
    "The match was thrilling and full of surprises.",# Sports
    "AI will revolutionize healthcare.",             # Tech
    "This pizza tastes amazing!",                    # Health
    "The weather is gloomy and cold."                # News
]
labels_a = [0, 1, 2, 3, 0]  # Topic: [News, Sports, Tech, Health, News]
labels_b = [1, 0, 0, 0, 2]  # Sentiment: [Neutral, Positive, Positive, Positive, Negative]

dataset = DummyMultiTaskDataset(tokenizer, sentences, labels_a, labels_b)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = MultiTaskModel()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

topic_labels = ["News", "Sports", "Tech", "Health"]
sentiment_labels = ["Positive", "Neutral", "Negative"]

# Training Loop
model.train()
for epoch in range(10):
    print(f"\n===== Epoch {epoch + 1} =====")
    total_correct_a = 0
    total_correct_b = 0
    total_samples = 0

    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels_a = batch["label_a"]
        labels_b = batch["label_b"]
        texts = batch["text"]

        logits_a, logits_b = model(input_ids, attention_mask)

        # Predictions
        preds_a = torch.argmax(logits_a, dim=1)
        preds_b = torch.argmax(logits_b, dim=1)

        # Accuracy calculations
        correct_a = (preds_a == labels_a).sum().item()
        correct_b = (preds_b == labels_b).sum().item()
        total_correct_a += correct_a
        total_correct_b += correct_b
        total_samples += labels_a.size(0)

        # Logging predictions
        print("\nBatch Predictions:")
        for i in range(len(preds_a)):
            print(f"Sentence: \"{texts[i]}\"\n → Topic: {topic_labels[preds_a[i]]} (Label: {topic_labels[labels_a[i]]})"
                  f", Sentiment: {sentiment_labels[preds_b[i]]} (Label: {sentiment_labels[labels_b[i]]})")

        # Loss
        loss_a = loss_fn(logits_a, labels_a)
        loss_b = loss_fn(logits_b, labels_b)
        total_loss = loss_a + 0.5 * loss_b

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Loss A: {loss_a.item():.4f}, Loss B: {loss_b.item():.4f}, Total: {total_loss.item():.4f}")

    # Epoch-level accuracy
    acc_a = total_correct_a / total_samples
    acc_b = total_correct_b / total_samples
    print(f"\nEpoch {epoch + 1} Accuracy → Task A (Topic): {acc_a:.2%}, Task B (Sentiment): {acc_b:.2%}")
