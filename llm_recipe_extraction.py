from bs4 import BeautifulSoup
import os
import json 
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google Generative AI LangChain LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def clean_recipes(input_file: str) -> str:
    """Clean the recipe HTML file and return the cleaned text."""
    with open(input_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        soup = soup.get_text(separator=" ", strip=True)
        f.close()
    return soup

def process_recipe_using_llm(recipes: list) -> str:
    """Send the recipes to the LLM and get the response per the system instructions."""
    SYS_INSTRUCT = "You are receiving a list of extracted text from recipe HTML files. Please extract the name, ingredients, and directions from each recipe and return the extracted information in JSON format with the following keys: 'name', 'ingredients', and 'directions' where each recipe is its own dictionary."
    
    messages = [
        (
            "system", SYS_INSTRUCT,
        ),
        (
            "user", recipes,
        )
    ]
    
    msg = llm.invoke(messages)
    return msg.content

def convert_to_json(msg: str) -> list:
    """Convert the LLM response to JSON format."""
    json_string = re.search(r'```json\n(.*)\n```', msg, re.DOTALL)
    if json_string:
        json_string = json_string.group(1)
        try:
            recipe_data = json.loads(json_string)
            return recipe_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print("JSON not found in response.")
    return []

def save_json(data: list, output_file: str):
    """Save the JSON data to a file."""
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    recipes = []
    input_dir = 'Recipes/Recipes'
    output_file = 'recipes_llm.json'
    for filename in os.listdir(input_dir):
        if filename.endswith('.html'):
            input_file = os.path.join(input_dir, filename)
            cleaned_text = clean_recipes(input_file)
            recipes.append(cleaned_text)
    recipes_chunk = []
    all_recipes = []
    # Process the recipes in chunks of 5 to avoid hitting token or query limits
    for i, recipe in enumerate(recipes):
        n = i + 1
        recipes_chunk.append(recipe)
        if n % 5 == 0:
            msg = process_recipe_using_llm(recipes_chunk)
            all_recipes.extend(convert_to_json(msg))
            recipes_chunk = []
        elif n == len(recipes):
            msg = process_recipe_using_llm(recipes_chunk)
            all_recipes.extend(convert_to_json(msg))
    save_json(all_recipes, output_file)

if __name__ == "__main__":
    main()