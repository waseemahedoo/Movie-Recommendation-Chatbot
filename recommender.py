import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, df_movies, embeddings):
        self.df = df_movies
        self.embeddings = embeddings

    def recommend(self, query_embedding, top_k=5):
        sims = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        return self.df.iloc[top_indices], sims[top_indices]
