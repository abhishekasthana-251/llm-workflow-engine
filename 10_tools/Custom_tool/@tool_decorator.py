from langchain_core.tools import tool

# #step1 -> create a function
# def multiply(a,b):
#     """MUltiply two numbers """
#     return a*b

# #step 2 -> add type hints

# def multiply(a:int, b:int)-> int:
#     """ Mulitiply two numbers"""
#     return a*b

#add tool decorator

@tool
def multiply(a:int, b:int)->int:
    """Mulitply two number"""  #lmm read this docstring
    return a*b

result=multiply.invoke({"a":25, "b":7})

print(result ,"\n")

print("\n",multiply.name)
print("\n",multiply.description)
print(multiply.args)