from transformers import AutoTokenizer, AutoModel
import torch

model_name = "osiria/distilbert-base-italian-cased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)

texts = [
    "Impossibile accedere al server FTP",
    "Timeout sulla richiesta HTTP"
]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs    = model(**encodings)
    embeddings = outputs.last_hidden_state[:, 0, :]

print("Shape embeddings:", embeddings.shape)  # → torch.Size([2, 768])
