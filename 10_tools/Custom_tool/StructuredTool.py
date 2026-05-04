#StructuredTool(when you need strict input vaildation)

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="1st number")
    b: int = Field(description="2nd number")

def multiply_fun(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply_fun,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a': 3, 'b': 3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)