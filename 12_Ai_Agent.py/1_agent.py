# this agent is normal just search on web about the the query

from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
#from langchain.agents import create_react_agent, AgentExecutor
#from langchain.agents import create_react_agent, AgentExecutor  # ✅
#from langchain_core.agents import create_react_agent
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_core.agents import create_react_agent  # ← moved here in 1.x

load_dotenv()


search_tool=DuckDuckGoSearchRun()

llm=ChatGroq(model="llama-3.3-70b-versatile")

#now the agent work is starting 

#-> pull the ReAct prompt from Langchain Hub

prompt= hub.pull("hwchase17/react") # pulls the standard ReAct agent Prompt

#-> Create the ReAct agent manually with the pulled prompt

agent=create_react_agent( # the agent need llm tools and prompt 
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

#-> AgentExecutor -> the agent think and AgentExecutor work on that, as agent do the office work and AgentExecutor do the field work

#wrap  it with AgentExecutor
agent_executor=AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True ,#verbose=True mean that in the output be get what the agent is thinking while give the output or executing the task
    handle_parsing_errors=True #prevents random crashes
)

response=agent_executor.invoke({"input":"3 ways to reach prayagraj from pune"})
print(response)