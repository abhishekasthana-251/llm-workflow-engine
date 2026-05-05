# creating a small project where we are converting the  currency of countrys 

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
import requests
import json

load_dotenv()

#creating a tool

@tool
def get_conversion_factor(base_currency: str, target_currency: str)-> float:
    """
    This function fetches the currency conversion factor between a given base currency and target currency

    """
    url=f"https://v6.exchangerate-api.com/v6/926ad32039d2756b8ff416ae/pair/{base_currency}/{target_currency}"

    response=requests.get(url)

    return response.json()

@tool
def convert(base_currency_value:int, conversion_rate:Annotated[float,InjectedToolArg])->float:
    """ 
    given a currency conversoion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency_value*conversion_rate
#conversion_rate:Annotated[float,InjectedToolArg] -> LLM do not try to fill this argument , I(the developer will inject this value after running earlier tools )




# print(get_conversion_factor.invoke({"base_currency":"USD","target_currency":"INR"}))

# ->{'result': 'success', 'documentation': 'https://www.exchangerate-api.com/docs', 'terms_of_use': 'https://www.exchangerate-api.com/terms', 'time_last_update_unix': 1777939201, 'time_last_update_utc': 'Tue, 05 May 2026 00:00:01 +0000', 'time_next_update_unix': 1778025601, 'time_next_update_utc': 'Wed, 06 May 2026 00:00:01 +0000', 'base_code': 'USD', 'target_code': 'INR', 'conversion_rate': 95.228}


# print("\n------------\n" \
# )
# print(convert.invoke({'base_currency_value':10,'conversion_rate':95.228})) ->952.28


# tool binding
llm=ChatGroq(model="llama-3.3-70b-versatile")

llm_with_tool=llm.bind_tools([get_conversion_factor,convert])

message=[HumanMessage("What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd")]

ai_message=llm_with_tool.invoke(message)

message.append(ai_message)

#ai_message.tool_calls
# [{'name': 'get_conversion_factor',
#   'args': {'base_currency': 'INR', 'target_currency': 'USD'},
#   'id': 'call_PKL8v7zwmphzNel0MnvjjGvY',
#   'type': 'tool_call'},
#  {'name': 'convert',
#   'args': {'base_currency_value': 10},
#   'id': 'call_vRdld30yHTKFGcTQPumgzH5u',
#   'type': 'tool_call'}]


for tool_call in ai_message.tool_calls:
    #execute the 1st tool and get the value of coversion rate
    if tool_call['name']=='get_conversion_factor':

        tool_message1=get_conversion_factor.invoke(tool_call)
        #fetch this conversion rate from the tool_message1 , but the tool_message1 is in the json
        conversion_rate=json.loads(tool_message1.content)['conversion_rate']
        #append this tool message to message list
        message.append(tool_message1)
    #execute the 2nd tool using the conversion rate from tool 1
    if tool_call['name']=='convert':
        #fetch the current arg , and also providing the converstion rate 
        tool_call['args']['conversion_rate']=conversion_rate

        tool_message2=convert.invoke(tool_call)
        message.append(tool_message2)

#print(message)

# [HumanMessage(content='What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd', additional_kwargs={}, response_metadata={}), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'dzx3bq3v9', 'function': {'arguments': '{"base_currency":"INR","target_currency":"USD"}', 'name': 'get_conversion_factor'}, 'type': 'function'}, {'id': '4v20q57xz', 'function': {'arguments': '{"base_currency_value":10}', 'name': 'convert'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 38, 'prompt_tokens': 352, 'total_tokens': 390, 'completion_time': 0.080084045, 'completion_tokens_details': None, 'prompt_time': 0.018154011, 'prompt_tokens_details': None, 'queue_time': 0.160660738, 'total_time': 0.098238056}, 'model_name': 'llama-3.3-70b-versatile', 'system_fingerprint': 'fp_4f6d808339', 'service_tier': 'on_demand', 'finish_reason': 'tool_calls', 'logprobs': None, 'model_provider': 'groq'}, id='lc_run--019df64e-da8a-79a3-8028-99a8337b635e-0', tool_calls=[{'name': 'get_conversion_factor', 'args': {'base_currency': 'INR', 'target_currency': 'USD'}, 'id': 'dzx3bq3v9', 'type': 'tool_call'}, {'name': 'convert', 'args': {'base_currency_value': 10, 'conversion_rate': 0.0105}, 'id': '4v20q57xz', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 352, 'output_tokens': 38, 'total_tokens': 390}), ToolMessage(content='{"result": "success", "documentation": "https://www.exchangerate-api.com/docs", "terms_of_use": "https://www.exchangerate-api.com/terms", "time_last_update_unix": 1777939201, "time_last_update_utc": "Tue, 05 May 2026 00:00:01 +0000", "time_next_update_unix": 1778025601, "time_next_update_utc": "Wed, 06 May 2026 00:00:01 +0000", "base_code": "INR", "target_code": "USD", "conversion_rate": 0.0105}', name='get_conversion_factor', tool_call_id='dzx3bq3v9'), ToolMessage(content='0.10500000000000001', name='convert', tool_call_id='4v20q57xz')]


print(llm_with_tool.invoke(message).content)