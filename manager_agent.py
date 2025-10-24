
import billing_agent
import sentiment_agent
# import explanation_agent

from openai import OpenAI  # or whichever client you use
import os
from dotenv import load_dotenv
from agents import Agent, Runner
import asyncio

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)



class Manager_Agent:
    def __init__(self):
        self.manager_agent = Agent(
            name="Manager agent",
            instructions=(
                "Handle all direct user communication. The goal is to help customers figure out their electricity bill" \
                "information and answer any questions they may have."
                "Call the relevant tools when specialized expertise is needed."
            ),  
            tools=[
                billing_agent.get_agent().as_tool(
                    tool_name="billing_expert",
                    tool_description="Handles bill retrieval and upload.",
                ),
                sentiment_agent.get_agent().as_tool(
                    tool_name="sentiment_expert",
                    tool_description="Handles judging question sentiment.",
                )
            ],
        )
      
            # handoffs = [billing_agent.get_agent(),
            #             sentiment_agent.get_agent()
            # ],
    async def run(self, query):    
        result = await Runner.run(self.manager_agent, query)
        # print(result.final_output)
        return result.final_output


    def handle_query(self, user_query: str, user_name: str = None) -> dict:
        full_query = user_name + ': ' + user_query
        result = {}
        result['response'] = asyncio.run(self.run(full_query))
        result['explanation'] = result['response']
        result['sentiment'] = result['response']
        return result


