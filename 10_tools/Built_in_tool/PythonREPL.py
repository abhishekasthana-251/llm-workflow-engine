#PythonREPLTool -> llm write and runs code
from langchain_experimental.tools import PythonREPLTool
repl=PythonREPLTool()
print(repl.run("print(2**10)"))



# repl.run("print(2**10)")
#         ↓
# PythonREPLTool spun up a Python interpreter
#         ↓
# Executed the string as real Python code
#         ↓
# Captured stdout → returned "1024"


