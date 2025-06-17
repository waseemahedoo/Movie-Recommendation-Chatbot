import streamlit as st
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from recommender import MovieRecommender
from llm_utils import rewrite_query_with_llama

torch.classes.__path__ = []

# Load movie data
df_movies = pd.read_csv("data/movies.csv")
descriptions = df_movies["Description"].tolist()

# Load precomputed description embeddings
embeddings_tensor = torch.load("data/movie_embeddings.pt")  
embeddings = embeddings_tensor.cpu().numpy()

# Load the embedding model
embedder = SentenceTransformer("intfloat/e5-base")

# Instantiate recommender
recommender = MovieRecommender(df_movies, embeddings)

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 LLaMA Movie Recommender Chatbot")

print("✅ App started")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    print("✅ Chat history initialized")


user_input = st.chat_input("What kind of movie are you looking for?")

if user_input:
    print(f"📝 User input: {user_input}")
    st.session_state.chat_history.append(user_input)

    # Rewriting user intent
    print("⚙️ Calling LLaMA to rewrite query...")
    rewritten_query = rewrite_query_with_llama(st.session_state.chat_history)
    print(f"🧠 Rewritten query: {rewritten_query}")
    st.markdown(f"🔍 **Searching for:** _{rewritten_query}_")

    # Embedding
    print("📦 Embedding rewritten query...")
    query_embedding = embedder.encode(rewritten_query)
    print(f"📐 Embedding shape: {query_embedding.shape}")

    # Recommend
    print("🔍 Retrieving top movie matches...")
    top_movies, scores = recommender.recommend(query_embedding)
    print(f"🎯 Top {len(scores)} results fetched")

    # Display results
    with st.chat_message("assistant"):
        for i, (_, row) in enumerate(top_movies.iterrows()):
            st.markdown(f"🎥 **{row['Movie Name']}**\n{row['Description']}\n")
            print(f"📽️ Movie #{i+1}: {row['Movie Name']}")

    with st.chat_message("user"):
        st.markdown(user_input)