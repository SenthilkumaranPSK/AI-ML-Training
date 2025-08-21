from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

faq_sentences = [
    "How can I reset my password?",
    "Where is the library located?",
    "What is Artificial Intelligence?",
    "How to apply for a scholarship?",
    "What are the cafeteria opening hours?"
]

def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

faq_embeddings = [get_embedding(sent) for sent in faq_sentences]

query = "what is AI"
query_embedding = get_embedding(query)
similarities = [cosine_similarity(query_embedding, emb)[0][0] for emb in faq_embeddings]
best_match_index = similarities.index(max(similarities))

print(f"Student Question: {query}")
print(f"Most Similar FAQ: {faq_sentences[best_match_index]}")
