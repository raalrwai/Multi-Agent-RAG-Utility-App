from agents import function_tool
from openai import OpenAI
import json, os
from dotenv import load_dotenv
import rag  

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_bills(name: str):
    return rag.retrieve_bill_embeddings(name)

def upload_bill(bill):
    return rag.file_to_upsert(bill)

def upload_history(history):
    return rag.history_to_file(history)

func_dict = {
    "get_bills": get_bills,
    "upload_bill": upload_bill,
    "upload_history": upload_history
}

input_list = [{"role": "system", "content": (
    "You are an agent that manages a database of bills: uploading, retrieving, "
    "and storing chat history. Return retrieved bills if already present."
)}]

@function_tool
def billing_agent(query: str, model: str = "gpt-5-chat-latest") -> str:
    """
    Determines which billing function to call, executes it, and returns LLM output.
    """
    input_list.append({"role": "user", "content": query})

    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "function",
                "name": "get_bills",
                "description": "Retrieve bills for a given person.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Name on bills"}},
                    "required": ["name"],
                },
            },
            {
                "type": "function",
                "name": "upload_bill",
                "description": "Upload a bill to the database",
                "parameters": {"type": "object", "properties": {"bill": {"type": "string"}}, "required": ["bill"]},
            },
            {
                "type": "function",
                "name": "upload_history",
                "description": "Upload chat history",
                "parameters": {"type": "object", "properties": {"history": {"type": "string"}}, "required": ["history"]},
            },
        ],
        input=input_list
    )

    for item in response.output:
        if item.type == "function_call" and item.name in func_dict:
            func = func_dict[item.name]
            func_result = func(**json.loads(item.arguments))
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"result": func_result})
            })

    final_response = client.responses.create(
        model=model,
        instructions=query,
        tools=[], 
        input=input_list
    )

    return final_response.output_text

BillingTool = billing_agent
