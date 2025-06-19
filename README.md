# 🎬 Movie Recommender Chatbot

A conversational movie recommender system that combines **SentenceTransformer embeddings**, **LLM-powered query rewriting**, and a **Streamlit chatbot interface** to provide relevant movie suggestions — based on user intent, mood, and multi-turn queries.

> 🧠 Powered by Mistral-7B-Instruct (via `transformers`) + `intfloat/e5-base` for semantic similarity

---

## 🛠️ Features

- ✅ **LLM-based query understanding**  
  Uses Mistral 7B to rewrite vague or complex user messages into focused movie search queries.

- ✅ **Semantic search with vector embeddings**  
  Movie descriptions are embedded using `intfloat/e5-base` and compared to user intent via cosine similarity.

- ✅ **Multi-turn conversation awareness**  
  The chatbot remembers recent user queries and rewrites them in context (e.g. "I want something sad but inspiring").

- ✅ **Streamlit chatbot UI**  
  Clean and interactive front-end to simulate a real-time recommendation dialogue.

---

## 🚀 Tech Stack

| Component     | Tool                                      |
|---------------|-------------------------------------------|
| Embeddings    | `intfloat/e5-base` (SentenceTransformer)  |
| LLM Rewriting | `mistralai/Mistral-7B-Instruct-v0.1`      |
| UI            | Streamlit (`st.chat_input`, `st.chat_message`) |
| Backend       | Python + Sklearn + Transformers           |

---

## 💡 Example Interaction

User: I want a romantic movie with a twist

User: Something involving social media

🧠 Rewritten Query: romantic movie with a twist involving social media

🎬 Recommended Movies:

Her — A man falls in love with an AI operating system.

The Circle — A woman joins a tech giant with hidden motives.

Searching — A father uncovers secrets online while looking for his missing daughter.
