from pydantic import BaseModel ,EmailStr,Field
from typing import Optional


class Student(BaseModel):

    name: str="Default Value"
    age : Optional[int]=None
    email : EmailStr

    # Feild is used for applying constraint
    cgpa : float =Field(gt=0,lt=10, description='A decimal value representing the cgpa of the student.') #gt=greater then & lt=lesser then


new_student ={'name':'shivam','age':25,'email':'shivam@gmail.com','cgpa':8.5,}

student = Student(**new_student)
print(student)


# can be converted into dictonary
student_dict = dict(student)
print(student_dict['age'])

# can be converted into json
student_json = student.model_dump_json




