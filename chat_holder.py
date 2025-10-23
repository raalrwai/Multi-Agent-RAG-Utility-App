# A MessagesPlaceholder in LangChain is a special placeholder used inside a ChatPrompTemplate to dynamically insert chat history or a list of messages at runtime.
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

load_dotenv()
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.5,max_completion_tokens=200)

# Chat Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name = 'chat_history'),
    ('human', '{query}')
])


chat_history = []
try:
    with open('4.LangChain_Prompts/11.chat_history.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("User: "):
                chat_history.append(HumanMessage(content=line.replace("User: ", "").strip()))
            elif line.startswith("AI: "):
                chat_history.append(AIMessage(content=line.replace("AI: ", "").strip()))
except FileNotFoundError:
    pass


# load Chat History
#chat_history = []
#with open('4.LangChain_Prompts/11.chat_history.txt') as f:
#    chat_history.extend(f.readlines())

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        print("Thanks for your time. See You!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:", result.content)


#print(chat_history)


with open('4.LangChain_Prompts/11.chat_history.txt', 'w', encoding='utf-8') as f:
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                f.write(f"User: {msg.content}\n")
            elif isinstance(msg, AIMessage):
                f.write(f"AI: {msg.content}\n")