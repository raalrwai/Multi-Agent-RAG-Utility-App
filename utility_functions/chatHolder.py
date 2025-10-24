from our_agents import function_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessage

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.5, max_completion_tokens=200)
chat_history = []

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

@function_tool
def langchain_chat(query: str) -> str:
    """
    Multi-turn LangChain chat interface.

    Args:
        query: user message string

    Returns:
        AI response string
    """
    chat_history.append(HumanMessage(content=query))

    response = model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    return response.content

LangChainChatTool = langchain_chat
