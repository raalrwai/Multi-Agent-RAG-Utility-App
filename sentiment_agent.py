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
            "In charge of judging a query's emotional sentiment in terms of positive or negative."
        ),
        handoff_description="Judges the sentiment value of a query."
    )
    return sentiment_agent

