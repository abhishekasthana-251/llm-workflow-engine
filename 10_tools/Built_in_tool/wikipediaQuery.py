#WikipediaQueryRun-> clean Factual Lookups
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wiki=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
print(wiki.run("Transformer neural network"))