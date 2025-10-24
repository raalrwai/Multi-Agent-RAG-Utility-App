import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner

import our_agents.billing_agent as billing_agent
import our_agents.sentiment_agent as sentiment_agent
import our_agents.explanation_agent as explanation_agent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class Manager_Agent:
    def __init__(self):
        """
        The manager acts as a coordinator. It decides when to call
        specialized agents (billing, sentiment, explanation).
        """
        self.manager_agent = Agent(
            name="Manager agent",
            instructions=(
                "You are the manager agent overseeing user interactions."
                "Your job is to interpret what the user wants, route queries "
                "to the appropriate specialist agents (billing, explanation, or sentiment), "
                "and return a helpful final answer."
            ),
            tools=[
                billing_agent.get_agent().as_tool(
                    tool_name="billing_expert",
                    tool_description="Handles retrieving and uploading electricity bills."
                ),
                sentiment_agent.get_agent().as_tool(
                    tool_name="sentiment_expert",
                    tool_description="Analyzes the user's tone and sentiment."
                ),
                explanation_agent.get_agent().as_tool(
                    tool_name="explanation_expert",
                    tool_description="Provides detailed explanations or general conversation replies."
                ),
            ],
        )

    async def run(self, query):
        result = await Runner.run(self.manager_agent, query)
        return result.final_output

    def handle_query(self, user_query: str, user_name: str = None, has_bill: bool = False) -> dict:
        """
        Handle an incoming query by delegating to the right agent.
        Explanation agent will always give the final response.
        """
        full_query = f"{user_name or 'User'}: {user_query}"

        billing_keywords = ["bill", "amount", "usage", "charge", "due date", "balance"]
        is_billing_related = any(kw in user_query.lower() for kw in billing_keywords)

        result = {}

        if is_billing_related:
            bill_response = asyncio.run(billing_agent.get_info(user_name, user_query))

            explanation_prompt = (
                f"The user asked about their bill.\n\n"
                f"Billing response: {bill_response}\n\n"
                f"Now explain this clearly to the user."
            )
            result["response"] = asyncio.run(
                explanation_agent.get_explanation(user_name, explanation_prompt)
            )
            result["source"] = "billing + explanation"

        else:
            if not has_bill:
                no_bill_message = (
                    "No bill has been uploaded yet. "
                    "You can upload a bill to get detailed explanations about your usage and charges."
                )
                explanation_prompt = f"{user_query}\n\n{no_bill_message}"
            else:
                explanation_prompt = user_query

            result["response"] = asyncio.run(
                explanation_agent.get_explanation(user_name, explanation_prompt)
            )
            result["source"] = "explanation"

        return result
