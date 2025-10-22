import os
from openai import OpenAI
from dotenv import load_dotenv

from rag import RAGAgent  
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


class SentimentAgent:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def analyze(self, text: str) -> str:
        """Classify text sentiment as positive, negative, or neutral."""
        prompt = f"Classify the sentiment of the following statement as Positive, Negative, or Neutral:\n\n{text}"
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        sentiment = response.choices[0].message.content.strip().lower()
        if "negative" in sentiment:
            return "negative"
        elif "positive" in sentiment:
            return "positive"
        else:
            return "neutral"


def get_response(system_prompt, user_prompt, model='gpt-5-chat-latest'):
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.output_text


class ExplanationAgent:
    def __init__(self, model="gpt-4"):
        self.model = model

    def explain(self, contexts: list, original_query: str) -> str:
        joined_contexts = "\n".join(contexts)
        prompt = (
            f"User asked: '{original_query}'\n"
            f"Please explain the relevant content from the following document snippets:\n\n{joined_contexts}"
        )
        return get_response(
            system_prompt="You are an assistant that explains document contents in clear and simple terms.",
            user_prompt=prompt,
            model=self.model
        )


class ManagerAgent:
    def __init__(self, sentiment_agent: SentimentAgent, rag_agent: RAGAgent, explanation_agent: ExplanationAgent):
        self.sentiment_agent = sentiment_agent
        self.rag_agent = rag_agent
        self.explanation_agent = explanation_agent

    def handle_query(self, user_query: str, user_name: str = None) -> dict:
        sentiment = self.sentiment_agent.analyze(user_query)

        contexts = self.rag_agent.retrieve(user_query)

        system_prompt = (
            "You are a helpful assistant that answers questions about electricity bills using relevant data."
        )
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


if __name__ == "__main__":
    sentiment_agent = SentimentAgent()
    rag_agent = RAGAgent()
    explanation_agent = ExplanationAgent()
    manager = ManagerAgent(sentiment_agent, rag_agent, explanation_agent)

    user_query = input("Ask a question about your electricity bill: ")
    result = manager.handle_query(user_query, user_name="Test User")

    print("\nAnswer:\n", result["response"])
    print("\nSentiment:", result["sentiment"])
    if result["explanation"]:
        print("\nExplanation:\n", result["explanation"])
