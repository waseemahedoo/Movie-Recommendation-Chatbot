from transformers import pipeline

def load_test_model():
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cpu",
        model_kwargs={"torch_dtype": "float32"}  # Use float32 for CPU
    )

def rewrite_query(chat_history, model=None):
    if model is None:
        model = load_test_model()

    prompt = "Chat history:\n"
    for msg in chat_history:
        prompt += f"User: {msg}\n"
    prompt += "\nTask: Rewrite the user's intent as a simple movie search query.\nAnswer:"

    response = model(prompt, max_new_tokens=50, return_full_text=False)
    rewritten = response[0]["generated_text"].strip()
    return rewritten

# === TEST CASE ===

def test_llm_rewrites_movie_query():
    chat_history = [
        "I want a romantic movie with a twist.",
        "Something involving social media."
    ]

    model = load_test_model()
    rewritten = rewrite_query(chat_history, model=model)

    print("🔁 Rewritten Query:", rewritten)

    assert isinstance(rewritten, str)
    assert "romantic" in rewritten.lower()
    assert "social media" in rewritten.lower()
    # assert len(rewritten.split()) < 20  # Should be a short query

if __name__ == "__main__":
    test_llm_rewrites_movie_query()

