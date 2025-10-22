# manager_agent.py

from rag import RAGAgent
from sentiment_agent import SentimentAgent
from explanation_agent import ExplanationAgent


class ManagerAgent:
    def __init__(self):
        self.sentiment_agent = SentimentAgent()
        self.rag_agent = RAGAgent()
        self.explanation_agent = ExplanationAgent()

    def handle_query(self, user_query: str, user_name: str = None) -> dict:
        sentiment = self.sentiment_agent.analyze(user_query)

        contexts = self.rag_agent.retrieve(user_query)

        system_prompt = "You are a helpful assistant that answers questions about electricity bills using relevant data."
        if sentiment == "negative":
            system_prompt += " The user seems upset. Respond with empathy and helpfulness."

        user_prompt = f"User query: {user_query}\n\nRelevant info:\n" + "\n".join(contexts)
        response = self.rag_agent.generate_response(system_prompt, user_prompt)

        needs_explanation = any(word in user_query.lower() for word in ["explain", "break down", "why", "how"])
        explanation = None
        if needs_explanation:
            explanation = self.explanation_agent.explain(contexts, user_query)

        return {
            "response": response,
            "sentiment": sentiment,
            "explanation": explanation
        }

