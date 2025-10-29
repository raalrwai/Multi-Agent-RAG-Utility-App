import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner # type: ignore

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
            model='gpt-4o-mini',
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

    sentiment_scores = []
    def get_average_sentiment_score(self):
        if self.sentiment_scores:
            return sum(self.sentiment_scores) / len(self.sentiment_scores)
        return 0.0

    async def run(self, query):
        result = await Runner.run(self.manager_agent, query)
        print('[manager] ', result.final_output, end='\n\n')
        return result.final_output

    async def run_manager_agent(self, query: str) -> str:
        result = await Runner.run(self.manager_agent, query)
        print('[manager] ', result.final_output, end='\n\n')
        return result.final_output

    async def handle_query(self, user_query: str, user_name: str = None, has_bill: bool = False, session=None) -> dict:
        full_query = f"{user_name or 'User'}: {user_query}"
        billing_keywords = ["bill", "amount", "usage", "charge", "due date", "balance"]
        is_billing_related = any(kw in user_query.lower() for kw in billing_keywords)

        # Analyze sentiment synchronously because the existing function is sync.
        sentiment_result = sentiment_agent.analyze_sentiment_and_intent(user_query)
        sentiment = sentiment_result.get("sentiment", "neutral")
        intent = sentiment_result.get("intent", "unknown")
        score = sentiment_result.get("score", 0.0)
        self.sentiment_scores.append(score)
        print(f'[Sentiment Agent] Sentiment: {sentiment}, Score: {score:.3f}, Intent: {intent}')
        
        print('Sentiment scores so far:', self.sentiment_scores)
        avg_score = self.get_average_sentiment_score()
        print(f"[Avg_Score] Average Sentiment Score: {avg_score:.3f}")

        greeting_intents = [
        'greeting', 'greet', 'hello', 'hi', 'salutation', 'checking in', 'well-being', 'asking how you are', 'saying hi', 'friendly approach', 'welcoming', 'greeting the assistant']
        if any(intent.lower().startswith(greet) or intent.lower() == greet for greet in greeting_intents):
            greeting_reply = f"Hello {user_name or 'there'}! Thanks for your friendly message. How can I help you today?"
            await session.add_items([
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": greeting_reply}
            ])
            return {"response": greeting_reply, "source": "manager_greeting"}

        output = f'[Sentiment Agent] Sentiment: {sentiment}, Intent: {intent}'
        await session.add_items([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": output}
        ])

        result = {}

        if is_billing_related:
            bill_response = await billing_agent.get_info(user_name, user_query, session=session)

            explanation_prompt = (
                f"The user asked about their bill.\n\n"
                f"Billing response: {bill_response}\n\n"
                f"Sentiment noted: {sentiment}. Intent noted: {intent}\n\n"
                "Now explain this clearly to the user."
            )

            result["response"] = await explanation_agent.get_explanation(user_name, explanation_prompt, session=session)
            result["source"] = "billing + explanation + sentiment"

        else:
            if not has_bill:
                no_bill_message = "No bill uploaded yet. Upload to get detailed explanations."
                explanation_prompt = f"{user_query}\n\n{no_bill_message}\n\nSentiment noted: {sentiment}. Intent noted: {intent}"
            else:
                explanation_prompt = f"{user_query}\n\nSentiment noted: {sentiment}. Intent noted: {intent}"

            result["response"] = await explanation_agent.get_explanation(user_name, explanation_prompt, session=session)
            result["source"] = "explanation + sentiment"

        return result

