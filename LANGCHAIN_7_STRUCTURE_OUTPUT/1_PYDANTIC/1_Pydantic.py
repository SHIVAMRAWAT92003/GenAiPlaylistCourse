
from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):

# Setting the default value
    name: str ="Default Value"
    age : Optional[int]=None

    # can do impicit conversion by their understanding
    salary:Optional[int]=None


new_Student = {'age':23,'salary':'2000000000'}

student = Student(**new_Student)

print(student)
print(student.name)
print(student.age)

print(student.salary)
print("Salary is of type ",type(student.salary))


