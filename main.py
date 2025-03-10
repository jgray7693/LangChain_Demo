from fastapi import FastAPI, Request
import json
import os
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import llm_recipe_extraction

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")  

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize ChromaDB client
vector_store = Chroma(
    collection_name="recipes",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# Initialize Google Generative AI LangChain LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize session history
session_history: dict[str, list] = {}

# Define ingest function to populate the vector store
def ingest_recipes_from_json(input_file: str):
    """Ingest recipes from a JSON file into the database."""
    with open(input_file, "r", encoding="utf-8") as f:
        recipes = json.load(f)
        documents = []
        for recipe in recipes:
            recipe_name = recipe["name"]
            recipe_ingredients = recipe["ingredients"]
            recipe_directions = recipe["directions"]

            recipe_text = f"Name: {recipe_name}\nIngredients: {' '.join(recipe_ingredients)}\nDirections: {' '.join(recipe_directions)}"
            document = Document(
                page_content=recipe_text,
                id=recipe_name,
                metadata={"name": recipe_name, "ingredients": ' '.join(recipe_ingredients)}
            )
            documents.append(document)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents, ids=uuids)

# Populate the vector store with recipes if not already done
if not vector_store.get("recipes"):
    # if recipes_llm.json does not exist
    if not os.path.exists("recipes_llm.json"):
        llm_recipe_extraction.main()
    ingest_recipes_from_json("recipes_llm.json")

# Define pydantic chat model
class ChatRequest(BaseModel):
    session_id: str
    query: str

@app.get("/")
async def index(request: Request):
    """Render the index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
def recipe_chatbot(request: ChatRequest) -> dict:
    """User-facing chatbot function to interact with the recipe database and a human user."""
    if request.session_id in session_history:
        response = query_llm_for_choice(request.session_id, request.query)
        if 'recipe_description' in response:
            response = response.split(":", 1)[1].strip()
            clear_history()
        return {"response": response}
    SYS_INSTRUCT = "You are a helpful recipe chatbot. You will receive a query from the 'user'. If the 'user' query contains ingredients, you should respond in the format of 'ingredients_list: ingredient1, ingredient2, ...'. If the 'user' query contains dish names, you should respond in the format of 'dish_list: dish1, dish2, ...'. If you receive any other query from the 'user', you should respond conversationally in order provide assistance in getting ingredients or dish names from the user."

    messages = [
        ("system", SYS_INSTRUCT),
        ("user", f'user: {request.query}'),
    ]
    response = llm.invoke(messages).content

    if "ingredients_list" in response:
        ingredients = response.split(":")[1].strip().split(", ")
        results = query_vector_store(" ".join(ingredients))
        response = query_llm_from_vector_store(" ".join(results))
        session_history[request.session_id] = results
        return {"response": response}
    elif "dish_list" in response:
        dishes = response.split(":")[1].strip().split(", ")
        results = query_vector_store(" ".join(dishes))
        response = query_llm_from_vector_store(" ".join(results))
        session_history[request.session_id] = results
        return {"response": response}
    else:            
        return {"response": response}
    
def query_vector_store(query: str, k: int = 5) -> list:
    """Query the vector store for similar recipes."""
    results = vector_store.similarity_search(query, k=k)
    results = [result.page_content for result in results]
    return results
    
def query_llm_from_vector_store(recipes: str) -> str:
    """Query the LLM for recipes based on ingredients."""
    SYS_INSTRUCT = "You are a helpful recipe chatbot. You will receive a query from 'recipe'. This query will be a joined list of recipes each with name, ingredients, and instructions. You should respond conversationally with the list of the recipes in the query and a brief description of the recipes."
    messages = [
        ("system", SYS_INSTRUCT),
        ("user", f"recipes: {recipes}"),
    ]
    response = llm.invoke(messages).content
    return response

def query_llm_for_recipe(recipe: str) -> str:
    """Query the LLM for a specific recipe."""
    messages = [
        ("system", "You are a helpful recipe chatbot."),
        ("user", f"Please provide the recipe for: {recipe}"),
    ]
    response = llm.invoke(messages).content
    return response

def query_llm_for_choice(session_id: str, recipe: str) -> str:
    """Query the LLM for a specific recipe choice"""
    SYS_INSTRUCT = f"You are a helpful recipe chatbot. You will receive a query from 'user' naming a specific recipe. Using the input recipe name and the context, please provide the name, ingredients, and instructions about the chosen recipe in a nice, readable format. If the query is unrelated to a recipe in the context, you should gently guide the user to selecting a new recipe. Please ensure that your response starts with 'recipe_description:' if you are responding with a recipe description selected by the user so that I know the proper response is obtained."
    messages = [
        ("system", SYS_INSTRUCT),
        ("user", f"recipe: {recipe}, context: {session_history[session_id]}"),
    ]
    response = llm.invoke(messages).content
    return response

def clear_history():
    """Clear the session history."""
    session_history.clear()
