# Pydantic gives us additionals data validation feautres then typed_dictonary
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    


new_Student1 = {'name':"shivam"}

student1 = Student(**new_Student1)
print(student1)
print(type(student1))



# This will give error bc we have defined name as string in class

# new_Student2 = {'name':123}
# student2 = Student(**new_Student2)
# print(student2)
# print(type(student2))

