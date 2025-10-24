# from billing_agent import Billing_Agent
# from sentiment_agent import Sentiment_Agent
# from explanation_agent import Explanation_Agent
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
                "Handle all direct user communication."
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


# class Manager_Agent:
#     def __init__(self):
#         self.sentiment_agent = Sentiment_Agent()
#         self.billing_agent = Billing_Agent()
#         self.explanation_agent = Explanation_Agent()

#     def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]
#         )
#         return response.choices[0].message.content.strip()

#     def handle_query(self, user_query: str, user_name: str = None, document_uploaded: bool = False) -> dict:
#         # Primer for LLM to decide which sub-agents to call
#         system_prompt = (
#             "You are a manager agent responsible for deciding which of the following agents to call "
#             "based on the user's query and whether a document is available:\n"
#             "- SentimentAgent: analyzes user sentiment\n"
#             "- RAGAgent: retrieves and generates answers based on documents\n"
#             "- ExplanationAgent: explains retrieved contexts when needed\n\n"
#             "Based on the input, decide which agents to call and provide a structured plan."
#         )

#         # Combine all info frontend has given
#         user_prompt = (
#             f"User query: {user_query}\n"
#             f"User name: {user_name}\n"
#             f"Document uploaded: {document_uploaded}\n"
#             "Please respond with which agents to call and any special instructions."
#         )

#         # Ask LLM for orchestration instructions
#         orchestration_plan = self._call_llm(system_prompt, user_prompt)

#         # For now, naive parsing: just check keywords in orchestration_plan
#         sentiment = None
#         contexts = []
#         explanation = None
#         response = ""

#         if "sentiment" in orchestration_plan.lower():
#             sentiment = self.sentiment_agent.analyze(user_query)
#         if "rag" in orchestration_plan.lower() or "retrieve" in orchestration_plan.lower():
#             contexts = self.billing_agent.make_request(user_query)
#             system_prompt_answer = (
#                 "You are a helpful assistant that answers questions about electricity bills using relevant data."
#             )
#             if sentiment == "negative":
#                 system_prompt_answer += " The user seems upset. Respond with empathy and helpfulness."

#             user_prompt_answer = f"User query: {user_query}\n\nRelevant info:\n" + "\n".join(contexts)
#             response = self.billing_agent.make_request(system_prompt_answer, user_prompt_answer)

#         # Determine if explanation is needed (or based on orchestration plan)
#         if any(word in user_query.lower() for word in ["explain", "break down", "why", "how"]) or "explanation" in orchestration_plan.lower():
#             if contexts:
#                 explanation = self.explanation_agent.make_request(contexts, user_query)

#         return {
#             "response": response,
#             "sentiment": sentiment,
#             "explanation": explanation,
#             "orchestration_plan": orchestration_plan,  # For debug/inspection
#         }

# # manager_agent.py

# from rag import RAGAgent
# from sentiment_agent import SentimentAgent
# from explanation_agent import ExplanationAgent


# class ManagerAgent:
#     def __init__(self):
#         self.sentiment_agent = SentimentAgent()
#         self.rag_agent = RAGAgent()
#         self.explanation_agent = ExplanationAgent()

#     def handle_query(self, user_query: str, user_name: str = None) -> dict:
#         sentiment = self.sentiment_agent.analyze(user_query)

#         contexts = self.rag_agent.retrieve(user_query)

#         system_prompt = "You are a helpful assistant that answers questions about electricity bills using relevant data."
#         if sentiment == "negative":
#             system_prompt += " The user seems upset. Respond with empathy and helpfulness."

#         user_prompt = f"User query: {user_query}\n\nRelevant info:\n" + "\n".join(contexts)
#         response = self.rag_agent.generate_response(system_prompt, user_prompt)

#         needs_explanation = any(word in user_query.lower() for word in ["explain", "break down", "why", "how"])
#         explanation = None
#         if needs_explanation:
#             explanation = self.explanation_agent.explain(contexts, user_query)

#         return {
#             "response": response,
#             "sentiment": sentiment,
#             "explanation": explanation
#         }

