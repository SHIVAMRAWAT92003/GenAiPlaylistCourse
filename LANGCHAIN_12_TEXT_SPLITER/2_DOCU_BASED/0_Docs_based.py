from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text =""""

import random

# Define a class to represent a magical creature
class MagicalCreature:
    def __init__(self, name):
        self.name = name
        self.type = random.choice(['Dragon', 'Unicorn', 'Phoenix', 'Griffin'])
        self.power_level = random.randint(50, 100)
    
    def display_info(self):
        return f"{self.name} is a {self.type} with a power level of {self.power_level}!"

# Define a function to create a random magical creature
def create_random_creature():
    names = ['Fluffy', 'Sparkle', 'Inferno', 'Storm', 'Shadow']
    random_name = random.choice(names)
    creature = MagicalCreature(random_name)
    return creature

# Generate and display information about a random magical creature
if __name__ == "__main__":
    random_creature = create_random_creature()
    print(random_creature.display_info())


"""
spliter =RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)


chunks = spliter.split_text(text)

print(len(chunks))
print(chunks[1])





