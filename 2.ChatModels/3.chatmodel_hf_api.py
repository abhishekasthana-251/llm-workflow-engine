"""from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint 
#the endpoint is used when we using a api, for the huggingFace hub
from dotenv import  load_dotenv
import os

load_dotenv()

llm= HuggingFaceEndpoint (
    #the tinLlama is now a paid version
    #repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    #repo_id="HuggingFaceH4/zephyr-7b-beta",
   # repo_id="microsoft/Phi-3-mini-4k-instruct",
    repo_id="google/gemma-2-2b-it",

    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 

    #here mainly two think is share 
    #repo_id= which repo_id you want from huggingface hub

    #and the  second is task , which task does this llm do 
)

model=ChatHuggingFace(llm=llm)
result=model.invoke(" JAI HIND")
print(result.content)


"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=100,
    provider="auto",  # ← THIS was missing! Lets HF choose best provider
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("what is the border of state of indai, and with which country")
print(result.content)