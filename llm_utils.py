from transformers import pipeline
import torch

# Load LLaMA 3.1 Instruct model
llama_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

def rewrite_query_with_llama(chat_history):
    prompt = (
        "You are an intelligent assistant for a movie recommendation chatbot.\n"
        "Given the conversation below, rewrite the latest user message as a standalone movie search query.\n"
        "The rewritten query should resemble a query to match a movie description. For example if you have as raw query 'Give me a romantic movie with a twist', the rewritten query should highlight clearly the plot that the user want to watch.\n"
        "If the query is unrelated to finding movies (e.g., asking for trailers or directors), summarize that as a separate intent.\n\n"
        "Conversation:\n"
    )
    for msg in chat_history:
        prompt += f"User: {msg}\n"

    prompt += "\nRewrite the user's intent:\n"

    output = llama_pipeline(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.3,
        return_full_text=False,
    )

    rewritten = output[0]["generated_text"].strip()
    print("🧠 Rewritten query:", rewritten)
    return rewritten