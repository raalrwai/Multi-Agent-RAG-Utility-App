import os
from openai import OpenAI
from our_agents import Agent, function_tool, Runner, SQLiteSession
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

@function_tool
def greet_user(name: str) -> str:
    return f"Hello, {name}! Today is a great day."

agent = Agent(
    name="TestAgent",
    instructions="You are a helpful assistant.",
    model="gpt-5-nano",
    tools=[greet_user]
)

session = SQLiteSession("test_session")  

if __name__ == "__main__":
    result = Runner.run_sync(agent, "Alice", session=session)
    print("Agent Output 1:", result.final_output)

    result = Runner.run_sync(agent, "What do you just said to me?", session=session)
    print("Agent Output 2:", result.final_output)
