import os

import pandas as pd
import json

from dotenv import load_dotenv
from openai import OpenAI

import rag

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
# PINECONE_INDEX_NAME = "retrieval-augmented-generation"

# pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
# index = pc.Index(PINECONE_INDEX_NAME)


tools = [
    {
        "type": "function",
        "name": "get_bills",
        "description": "Retrieve bills from a database for a given person.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name that appears on the bills",
                },
            },
            "required": ["name"],
        },
    },    
]

def get_bills(name):
    return None



func_dict = {'': }


class Explanation_Agent():
    def __init__(self):
        system_instruction = {"role": "system", "content": "You are an agent whose role is to interpret bill "
                        "and provide purely factual numbers and statistics from that bill that are relevant to"
                        "a question."}
        self.input_list = [system_instruction]

    def make_request(self, query, model='gpt-5-chat-latest'):

        self.input_list += {"role": "user", "content": query}

        response = client.responses.create(
            model=model,
            tools=tools,
            input=self.input_list,
        )    
        
        self.input_list += response.output

        # original_stdout = sys.stdout
        # with open('temp.txt', 'w') as f:
        #     sys.stdout = f
        #     print(response)
        # sys.stdout = original_stdout

        for item in response.output:
            if item.type == "function_call":
                if item.name in func_dict.keys():
                    func = func_dict[item.name]                    
                    response = func(json.loads(item.arguments))
                    
                    self.input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({
                            "bill": response
                            })
                    })

        response = client.responses.create(
            model=model,
            instructions= query,
            tools=tools,
            input=self.input_list,
        )

        return response.output_text



# if __name__ == '__main__':
#     test_dir = os.path.join(os.getcwd(), 'tests/billing_agent/')
#     test_questions_df = pd.read_csv(os.path.join(test_dir, 'questions.csv'))

#     answers_path = os.path.join(test_dir, 'answers.csv')
#     with open(answers_path, 'w') as f:
#         for i in range(len(test_questions_df)):
#             name = test_questions_df.loc[i]['Name']
#             question = test_questions_df.loc[i]['Question']
#             answer = '"' + ask_gpt(name,question).replace('\n', ' ').replace('"', "'") +'"'
#             content = ','.join([name,question,answer]) + '\n'
#             f.write(content)
                


