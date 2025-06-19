from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
torch.classes.__path__ = []
from dotenv import load_dotenv
import os

load_dotenv() 


model_path = os.getenv('MODEL_PATH')

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",  # or float16 / float32 based on device
    device_map="auto"
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer
)

def rewrite_query_with_llama(chat_history):
    prompt = (
    "You are an intelligent assistant inside a movie recommendation chatbot.\n"
    "Your task is to rewrite the latest user message as a clean, standalone movie search query.\n\n"
    "Guidelines:\n"
    "- The query must describe the type of movie the user wants to watch.\n"
    "- Use concrete themes, tone, genre, and any emotions or twists mentioned.\n"
    "- DO NOT just repeat 'romantic movie'. Be descriptive and specific.\n"
    "- If the user is vague, combine their last few inputs to infer what they mean.\n"
    "- If the user asks for something other than a movie (e.g., trailer, actor), return:\n"
    "  [non-search intent: user asked for a trailer]\n\n"
    "Examples:\n"
    "User: I want something emotional and powerful.\n"
    "Rewritten: emotional drama about overcoming trauma or loss\n\n"
    "User: It's raining, give me a movie with a twist.\n"
    "Rewritten: romantic movie set during rain with an unexpected twist\n\n"
    "User: Can you show me a trailer?\n"
    "Rewritten: [non-search intent: user asked for a trailer]\n\n"
    "User: I want something fun, maybe friends or college.\n"
    "Rewritten: lighthearted comedy about friendship and college life\n\n"
    "Conversation:\n"
)
    for msg in chat_history:
        prompt += f"User: {msg}\n"

    prompt += "\nRewrite the user's intent:\n"

    output = llm(
        prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.3,
        return_full_text=False,
    )

    rewritten = output[0]["generated_text"].strip()
    print("🧠 Rewritten query:", rewritten)
    return rewritten

