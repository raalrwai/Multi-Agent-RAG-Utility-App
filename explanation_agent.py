import os
from openai import OpenAI
from agents import function_tool
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

@function_tool
def explain_bills(query: str, relevant_contexts: list[str] | None = None) -> str:
    """
    Generates explanation of electricity bills.
    """
    prompt = f"Answer the following question about electricity bills:\n{query}\n"
    if relevant_contexts:
        prompt += "\nRelevant info:\n" + "\n".join(relevant_contexts)

    response = client.responses.create(
        model="gpt-5-chat-latest",
        input=[
            {"role": "system", "content": "You are an assistant explaining electricity bills."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.output_text
