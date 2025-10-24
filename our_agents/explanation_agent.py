import os
import json
import pandas as pd
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


@function_tool
def explain_bill_details(name: str, question: str, relevant_contexts: list[str] | None = None) -> str:
    """
    Generates clear, factual explanations about the contents of a user's electricity bill.
    Args:
        name: The name on the electricity bill.
        question: The user's natural language question.
        relevant_contexts: Optional list of related context strings from retrieved documents.
    Returns:
        A detailed but factual explanation of the bill or question asked.
    """
    prompt = f"""
    The user's name is {name}.
    Question: {question}

    Please provide a clear, concise, and factual explanation about this bill.
    """
    if relevant_contexts:
        prompt += "\nRelevant context:\n" + "\n".join(relevant_contexts)

    response = client.responses.create(
        model="gpt-5-chat-latest",
        input=[
            {"role": "system", "content": "You are an expert at explaining electricity bills clearly and factually."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.output_text


def get_agent():
    """
    Creates and returns the Explanation Agent.
    """
    explanation_agent = Agent(
        name="Explanation agent",
        instructions=(
            "Provide detailed and factual explanations about electricity bills."
            "Help the user understand their charges, consumption, and other components."
        ),
        handoff_description="Specialist agent for explaining electricity bills.",
        tools=[explain_bill_details]
    )
    return explanation_agent


async def get_explanation(name: str, question: str):
    """
    Runs the Explanation Agent asynchronously to generate an explanation.
    """
    query = f'For the customer "{name}", explain the following question: {question}'
    result = await Runner.run(get_agent(), query)
    return result.final_output


if __name__ == "__main__":
    test_dir = os.path.join(os.getcwd(), "tests/explanation_agent/")
    os.makedirs(test_dir, exist_ok=True)
    test_questions_path = os.path.join(test_dir, "questions.csv")

    if os.path.exists(test_questions_path):
        test_questions_df = pd.read_csv(test_questions_path)
        answers_path = os.path.join(test_dir, "answers.csv")

        with open(answers_path, "w") as f:
            for i in range(len(test_questions_df)):
                name = test_questions_df.loc[i]["Name"]
                question = test_questions_df.loc[i]["Question"]
                response = asyncio.run(get_explanation(name, question))
                answer = '"' + response.replace("\n", " ").replace('"', "'") + '"'
                content = ",".join([name, question, answer]) + "\n"
                f.write(content)
