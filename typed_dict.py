#typed dict is a way to define dict in python  where you can specify  what key s and values should exist --> ensures dict follows specific structure


from langchain_ollama import ChatOllama
from typing import TypedDict


class Person(TypedDict): #inheriting the Typed dict class
    name: str
    age: int

new_person: Person = {"name": 'pranay', "age": 30}

print(new_person)