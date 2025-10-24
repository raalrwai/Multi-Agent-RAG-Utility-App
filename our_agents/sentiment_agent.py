import json
import os
from openai import OpenAI
from agents import Agent  # type: ignore

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def analyze_sentiment_and_intent(text: str) -> dict:
    print("[sentiment check]")
    
    prompt = (
        f"Analyze the following customer query. "
        f"Respond with the sentiment (positive, negative, neutral) and the main intent (what the customer is trying to achieve) "
        f"in JSON format.\n\nQuery: {text}\n\nFormat:\n{{\"sentiment\": \"sentiment_value\", \"intent\": \"customer_intent\"}}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    

    print('[sentiment] ', json.loads(content), end='\n\n')
    try:
        result = json.loads(content)
        if "sentiment" in result and "intent" in result:
            return result
        else:
            return {"sentiment": "neutral", "intent": "unknown"}
    except Exception:
        return {"sentiment": "neutral", "intent": "unknown"}


def get_agent():
    def sentiment_intent_tool(query: str) -> str:
        result = analyze_sentiment_and_intent(query)
        return json.dumps(result)

    sentiment_agent = Agent(
        name="Sentiment and Intent Agent",
        instructions="Judge customer's query's sentiment and intent.",
        tools=[sentiment_intent_tool],
        handoff_description="Handles sentiment and intent understanding of queries."
    )
    return sentiment_agent


# import os

# import pandas as pd
# import json

# from dotenv import load_dotenv
# from openai import OpenAI
# from agents import Agent # type: ignore
# import rag

# load_dotenv()

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# client = OpenAI(api_key=OPENAI_API_KEY)



# def get_agent():
#     sentiment_agent = Agent(
#         name="Sentiment agent",
#         instructions=(
#             "You are in charge of judging a query's emotional sentiment in terms of positive or negative."
#         ),
#         handoff_description="Judges the sentiment value of a query."
#     )
#     return sentiment_agent


# import json
# from openai import OpenAI
# from dotenv import load_dotenv
# from agents import function_tool

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

# @function_tool
# def analyze_sentiment(chat_content: str) -> dict:
#     """
#     Analyze chat content and classify customer satisfaction.
#     Returns a dictionary with 'sentiment' and 'reason'.
#     """
#     prompt = f"""
#     You are a sentiment analysis expert for customer service chats.
#     Analyze the following chat transcript and classify the customer's satisfaction as one of:
#     Satisfied, Neutral, Unsatisfied.
#     Respond in strict JSON format like:
#     {{ "sentiment": "<Satisfied/Neutral/Unsatisfied>", "reason": "<brief justification>" }}
    
#     Chat transcript:
#     {chat_content}
#     """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#     )

#     text = response.choices[0].message["content"]
#     try:
#         return json.loads(text)
#     except Exception:
#         return {"sentiment": None, "reason": text}

# SentimentTool = analyze_sentiment
