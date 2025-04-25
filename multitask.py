import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer



class MultiTaskModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes_a=4, num_classes_b=3):
        super(MultiTaskModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.3)

        # Task A: e.g., Topic classification (News, Sports, Tech, Health)
        self.task_a_head = nn.Linear(768, num_classes_a)

        # Task B: e.g., Sentiment analysis (Positive, Neutral, Negative)
        self.task_b_head = nn.Linear(768, num_classes_b)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        model_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(model_output, attention_mask)
        pooled_output = self.dropout(pooled_output)

        logits_a = self.task_a_head(pooled_output)  # Task A: Topic classification
        logits_b = self.task_b_head(pooled_output)  # Task B: Sentiment classification

        return logits_a, logits_b

model = MultiTaskModel()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentences = [
    "I love working with transformers.",
    "The weather today is gloomy.",
    "Machine learning is the future."
]

# Tokenize inputs
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Run model
logits_a, logits_b = model(inputs["input_ids"], inputs["attention_mask"])
# Simulated class names
topic_labels = ["News", "Sports", "Tech", "Health"]
sentiment_labels = ["Positive", "Neutral", "Negative"]

preds_a = torch.argmax(logits_a, dim=1)
preds_b = torch.argmax(logits_b, dim=1)

print("\nTask A Predictions (Topic):")
for i, pred in enumerate(preds_a):
    print(f"Sentence: {sentences[i]} --> Topic: {topic_labels[pred.item()]}")

print("\nTask B Predictions (Sentiment):")
for i, pred in enumerate(preds_b):
    print(f"Sentence: {sentences[i]} --> Sentiment: {sentiment_labels[pred.item()]}")

print("Logits for Task A (Topic):", logits_a)
print("Logits for Task B (Sentiment):", logits_b)
