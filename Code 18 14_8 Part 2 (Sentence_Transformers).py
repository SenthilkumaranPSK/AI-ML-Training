from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

model = SentenceTransformer("all-MiniLM-L6-v2")

test_cases = [
    ("MS Dhoni", "India"),
    ("MS Dhoni", "Australia"),
    ("Ricky Ponting", "Australia"),
    ("Virat Kohli", "India"),
    ("Kane Williamson", "New Zealand"),
    ("Joe Root", "India"),
]

players = [p for p, _ in test_cases]
countries = list(set([c for _, c in test_cases]))  
player_embs = model.encode(players, normalize_embeddings=True, batch_size=8)
country_embs = model.encode(countries, normalize_embeddings=True, batch_size=8)

sim_matrix = util.cos_sim(player_embs, country_embs).cpu().numpy()

threshold = 0.78  

results = []
for i, player in enumerate(players):
    sims = sim_matrix[i]
    best_idx = int(np.argmax(sims))
    best_country = countries[best_idx]
    best_score = sims[best_idx]
    label = "MATCH" if best_score >= threshold else "NO MATCH"
    
    results.append({
        "Player": player,
        "Actual Country": dict(test_cases)[player],  
        "Predicted Country": best_country,
        "Similarity": round(float(best_score), 3),
        "Label": label
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
