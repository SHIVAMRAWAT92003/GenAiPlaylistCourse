from typing import TypedDict

class Person(TypedDict):
    name:str
    age: int



newPerson : Person={"name":"Ravi", "age":22} 
print(newPerson)