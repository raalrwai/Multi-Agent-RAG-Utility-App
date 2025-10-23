from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Load the LLM
centiment = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def chat_sentiment(chat_file_path):
    """Analyze chat and classify customer satisfaction"""

    with open(chat_file_path, 'r', encoding='utf-8') as f:
        chat_history = f.read()
    
    system_prompt = SystemMessage(content="""
    You are a sentiment analysis expert trained for customer service chat evaluation.
    Analyze the full chat between the customer and the AI agent of Nexus Energy.
    Based on tone, language, and context, classify the customer's satisfaction as one of the following categories:

    1. Satisfied – positive or thankful tone, issue resolved, user shows appreciation or relief.
    2. Neutral – polite, factual, or short responses, no clear positive or negative emotion.
    3. Unsatisfied – frustrated, complaining, confused, angry, or expressing dissatisfaction.

    Respond in JSON format:
    {
      "sentiment": "<Satisfied / Neutral / Unsatisfied>",
      "reason": "<brief justification for your classification>"
    }
    """)
    user_prompt = HumanMessage(content=f"Here is the full chat transcript:\n\n{chat_history}\n\nPlease classify it.")
    analysis_response = centiment.invoke([system_prompt, user_prompt])
    return analysis_response.content

result = chat_sentiment("4.LangChain_Prompts/11.chat_history.txt")
print(result)
