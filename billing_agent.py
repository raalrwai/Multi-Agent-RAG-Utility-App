from agents import function_tool
from openai import OpenAI
import json, os
from dotenv import load_dotenv
from openai import OpenAI

import rag

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
    {
        "type": "function",
        "name": "upload_bill",
        "description": "Upload a given bill to the database",
        "parameters": {
            "type": "object",
            "properties": {
                "bill": {
                    "type": "string",
                    "description": "The bill to be uploaded",
                },
            
            },
            "required": ["bill"],
        },
    },  
    {
        "type": "function",
        "name": "upload_history",
        "description": "Uploads the chat history to the database for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "history": {
                    "type": "string",
                    "description": "A history log of a user conversation with our model.",
                },
            
            },
            "required": ["history"],
        },
    },    
]

def get_bills(name):
    # print('NAME: ', name)
    # print(name['name'])
    return rag.retrieve_bill_embeddings(name['name'])

def upload_bills(bill):
    return rag.file_to_upsert(bill)

# To be Made
def upload_history(history):
    return rag.history_to_upsert(history)



func_dict = {'get_bills': get_bills, 
             'upload_bill': upload_bills,
             'upload_history': upload_history
             }


class Billing_Agent():
    def __init__(self):
        system_instruction = {"role": "system", "content": "You are an agent whose role is to be in charge"
                        "of managing a database containing bills. This includes uploading and downloading"
                        "bills, and uploading the chat history once a conversation has been completed. If"
                        "a requested bill has already been retrieved, return it again instead of downloading"
                        "it again."}
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
                


