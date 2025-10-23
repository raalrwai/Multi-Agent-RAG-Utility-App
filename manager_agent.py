import os
import asyncio
from agents import Agent, function_tool, Runner, SQLiteSession
from rag import RAGAgent
from sentiment_agent import SentimentTool as LangChainSentimentTool
from explanation_agent import explain_bills  

rag_agent = RAGAgent()
sentiment_tool = LangChainSentimentTool  

@function_tool
def rag_retrieve(query: str):
    """Retrieve relevant contexts for a user query."""
    return rag_agent.retrieve(query)

@function_tool
def rag_generate(system_prompt: str, user_prompt: str):
    """Generate response using RAG with system/user prompts."""
    return rag_agent.generate_response(system_prompt, user_prompt)

explain = explain_bills  

manager_agent = Agent(
    name="ManagerAgent",
    instructions="You are a manager agent that decides which tools to call internally based on user input and context.",
    model="gpt-5-nano",
    tools=[rag_retrieve, rag_generate, explain],
)

manager_session = SQLiteSession("manager_session")


def safe_run_sync(tool, query):
    """Run Runner.run_sync() in its own event loop (for Streamlit safety)."""
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = Runner.run_sync(tool, query)
        loop.close()
        return result
    except Exception as e:
        print(f"[safe_run_sync error] {e}")
        return None


def run_manager_query(user_query: str, user_name: str = None, document_uploaded: bool = False) -> dict:
    """Handles sentiment + RAG retrieval/generation + explanation."""

    sentiment_result = asyncio.run(asyncio.to_thread(safe_run_sync, sentiment_tool, user_query))

    sentiment_data = sentiment_result.final_output if sentiment_result else {}
    sentiment = sentiment_data.get("sentiment")

    contexts = []
    response = ""

    if document_uploaded or "rag" in user_query.lower():
        contexts = rag_retrieve.fn(user_query)
        system_prompt_answer = "You are a helpful assistant for electricity bills."

        if sentiment in ["Unsatisfied", "negative"]:
            system_prompt_answer += " Respond with empathy and helpfulness."

        user_prompt_answer = f"User query: {user_query}\n\nRelevant info:\n" + "\n".join(contexts)
        response = rag_generate.fn(system_prompt_answer, user_prompt_answer)

    explanation_result = None
    if any(word in user_query.lower() for word in ["explain", "why", "how", "break down"]) and contexts:
        explanation_result = explain.fn(contexts, user_query)

    return {
        "response": response,
        "explanation": explanation_result,
        "orchestration_plan": user_query.lower(),
    }


if __name__ == "__main__":
    result = Runner.run_sync(manager_agent, "Why is my electricity bill so high?", session=manager_session)
    print(result.final_output)
