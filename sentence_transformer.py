import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json



class SentenceTransformer(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(SentenceTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode(self, sentences):
        encoded = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.transformer(**encoded)
        embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        return embeddings


if __name__ == "__main__":
    model = SentenceTransformer()
    sentences = [
        "Fetch is building something cool.",
        "I enjoy working on ML models.",
        "Transformers are powerful in NLP."
    ]
    embeddings = model.encode(sentences)
    print("Sentence Embeddings Shape:", embeddings.shape)
    print("Sample Embedding Vector (first sentence):")
    print(embeddings[0])

# Convert embeddings to a Python list
embeddings_list = embeddings.detach().numpy().tolist()

# Save to a JSON file
with open("sentence_embeddings.json", "w") as f:
    json.dump({
        "sentences": sentences,
        "embeddings": embeddings_list
    }, f, indent=2)

print("Embeddings exported to sentence_embeddings.json")
