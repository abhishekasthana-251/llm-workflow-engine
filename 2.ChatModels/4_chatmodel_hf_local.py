
# ============================================
# HuggingFace Pipeline - Local Model
# Downloads model to your laptop and runs it
# No API key needed! But needs good hardware
# ============================================

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# to store in d drive insted of c drive 
import os
os.environ['HF_HOME']='D:/Huggingfac_cache'

llm= HuggingFacePipeline.from_model_id(
    #model_id="deepseek-ai/DeepSeek-R1-0528",# to big file 600gb file
    model_id="HuggingFaceTB/SmolLM2-360M-Instruct", #720mb file
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=100
    )
)
model=ChatHuggingFace(llm=llm)

result=model.invoke("what is the full form of HRI in jhusi prayagraj india")

print(result)