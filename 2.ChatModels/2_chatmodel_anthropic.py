# it is also paid , so I write a code but is will not work 
# website-> console.anthropic.com
#create key , copy it and paste in .env file , and make ANTHROPIC_API_KEY

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model="claud-3-5-sonnet-20241022")

result=model.invoke("what is today date ")

print(result.content)


# same with the gemini 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1-5-pro")

result=model.invoke("how you are")

print(result)