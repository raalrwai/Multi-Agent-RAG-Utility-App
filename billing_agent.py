class ManagerAgent:
    def __init__(self, sentiment_agent, rag_agent, explanation_agent):
        self.sentiment_agent = sentiment_agent
        self.rag_agent = rag_agent
        self.explanation_agent = explanation_agent

    def handle(self, user_query: str, user_name: str = None, need_explanation: bool = False):
        """
        Coordinates the sentiment, retrieval, and explanation agents.
        """
        sentiment = self.sentiment_agent.analyze(user_query)

        rag_response = self.rag_agent.query(user_query, user_name=user_name, sentiment=sentiment)

        explanation = None
        if need_explanation:
            explanation = self.explanation_agent.explain(contexts=rag_response.get("contexts", []), query=user_query)

        return {
            "response": rag_response["response"],
            "sentiment": sentiment,
            "explanation": explanation
        }
