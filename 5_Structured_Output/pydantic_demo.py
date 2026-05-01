from pydantic import BaseModel

class Student(BaseModel):
    name:str

new_student={'name':'Abhishek'}


student=Student(**new_student)

print(student)