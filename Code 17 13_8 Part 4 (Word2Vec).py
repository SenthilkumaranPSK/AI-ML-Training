from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sits", "on", "the", "mat"],
    ["the", "dog", "sits", "on", "the", "sofa"],
    ["dogs", "and", "cats", "are", "good", "pets"],
    ["i", "love", "my", "pet", "cat"]
]

model = Word2Vec(
    sentences=sentences,
    vector_size=50, 
    window=2,
    min_count=1,
    sg=1,             
    seed=42           
)

print("Vector for 'cat':")
print(model.wv['cat'])

print("\nMost similar to 'cat':")
for word, similarity in model.wv.most_similar('cat', topn=3):
    print(f"{word}: {similarity:.3f}")
