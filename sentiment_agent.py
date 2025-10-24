import os

import pandas as pd
import json

from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent

import rag

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)



def get_agent():
    sentiment_agent = Agent(
        name="Sentiment agent",
        instructions=(
            "You are in charge of judging a query's emotional sentiment in terms of positive or negative."
        ),
        handoff_description="Judges the sentiment value of a query."
    )
    return sentiment_agent

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
