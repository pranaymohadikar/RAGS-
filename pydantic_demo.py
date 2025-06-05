from pydantic import BaseModel, EmailStr
from typing import Optional


# class Student(BaseModel):
#     name: str = "Pranay"
    
# new_student = {}

# student = Student(**new_student)

# print(student)


# here name is string but if we out any otehr datatype, it will validate and give error as it is defined as string not otehr data type, but that previous one can happen in the typed dict as it wont validate tje datatype

#we can give default values calso by providin name iin the class


# we can set the optional values a;so as sometimes teh data wont be there

class Student(BaseModel):
    name: str = "Pranay"
    age: Optional[int]= None
    email: EmailStr
    
new_student = {"age": 30, "email":'abc@abc.com'}

student = Student(**new_student)

print(student)

student_json = student.model_dump_json()


#it has email validator also need to import Emailstr
