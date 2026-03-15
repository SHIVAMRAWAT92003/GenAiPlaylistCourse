
from pydantic import BaseModel,EmailStr

class Student(BaseModel):

# Setting the default value
    name: str ="Default Value"
    email: EmailStr #perform email validation


new_Student1 = {'email':'abc@gmail.com'}
student1 = Student(**new_Student1)
print(student1)


# This will give me error b/c email should have @gmail.com in end.
# new_Student2 = {'email':'abc'}
# student2 = Student(**new_Student2)
# print(student2)


