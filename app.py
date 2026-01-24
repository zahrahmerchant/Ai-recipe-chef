import streamlit as st
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import chromadb
import json
import os
import random
import time
import markdown

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "indian_recipes"
RECIPE_DATA_FILE = "recipes.json"
# Updated to use the standard flash model name, or your specific preview version
GENERATION_MODEL_NAME = "gemini-2.0-flash" 

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Recipe Chef Pro",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLES ---
st.markdown(
    """
<style>
    .main-header { font-size: 3rem; color: #FF4B4B; text-align: center; font-weight: 800; margin-bottom: 0; }
    .sub-header { font-size: 1.2rem; color: gray; text-align: center; margin-bottom: 2rem; font-style: italic; }
    .recipe-container { background-color: #ffffff; padding: 40px; border-radius: 15px; border: 1px solid #e0e0e0; box-shadow: 0 10px 25px rgba(0,0,0,0.08); margin: 20px auto 40px auto; max-width: 1000px; color: #333333; }
    .recipe-container h1 { color: #c0392b !important; font-family: 'Georgia', serif !important; font-size: 2.5rem !important; border-bottom: 3px solid #e74c3c !important; }
    .recipe-container h2 { color: #e67e22 !important; border-bottom: 1px solid #eee !important; }
    .recipe-container-long { max-height: 700px; overflow-y: auto; }
</style>
""",
    unsafe_allow_html=True,
)

# --- GEMINI API SETUP ---
# Ensure "GEMINI_API_KEY" is set in your Streamlit Cloud Secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"Error initializing Gemini Client. Check Secrets. {e}")
    st.stop()

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are "AI Recipe Chef," an expert Michelin-star Indian chef. 
Your goal is to create a NEW, creative, and delicious recipe based on the user's ingredients.

RULES:
1. Use the "Context Recipes" for inspiration, but DO NOT copy them.
2. Adhere strictly to dietary preferences and meal type.
3. Title MUST start with a single hash (# Title).
4. Use double hash (##) for "Ingredients" and "Instructions".
"""

# --- CACHED RESOURCES ---
@st.cache_resource
def get_resources():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    try:
        col = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        if col.count() > 0:
            first_item = col.peek(1)
            if 'full_text' not in first_item['metadatas'][0]:
                raise ValueError("Schema mismatch")
    except Exception:
        try:
            chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except:
            pass

    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    if collection.count() == 0:
        populate_database(collection, embedding_model)

    return embedding_model, collection


@st.cache_data
def load_recipe_data():
    with open(RECIPE_DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def populate_database(collection, model):
    try:
        recipes = load_recipe_data()
        documents, metadatas, ids = [], [], []

        for i, recipe in enumerate(recipes):
            doc_text = f"{recipe.get('title', '')} {recipe.get('ingredients', '')}"
            documents.append(doc_text)
            metadatas.append({
                "title": recipe.get('title', ''),
                "full_text": f"Title: {recipe.get('title', '')}\n\nIngredients:\n{recipe.get('ingredients', '')}\n\nInstructions:\n{recipe.get('instructions', '')}",
            })
            ids.append(str(i))

        embeddings = model.encode(documents).tolist()
        collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
    except Exception as e:
        st.error(f"Error populating database: {e}")
        st.stop()


# --- CORE HELPERS ---
def retrieve_recipes(query, k=3):
    embedding_model, collection = get_resources()
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k, include=["metadatas"])

    context_text = ""
    recipe_names = []
    if results and results['metadatas']:
        for i, meta in enumerate(results['metadatas'][0]):
            context_text += f"--- Context {i+1}: {meta['title']} ---\n{meta['full_text']}\n\n"
            recipe_names.append(meta['title'])
    return context_text, recipe_names


def generate_recipe(ingredients, preferences, meal_type, creativity, context):
    prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST:\n- Ingredients: {ingredients}\n- Preferences: {preferences}\n- Meal: {meal_type}\n\nCONTEXT:\n{context}"
    temp = 0.2 + (creativity * 0.08)

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temp)
        )
        return response.text if response.text else "Generation failed."
    except Exception as e:
        return f"Error: {e}"


def generate_nutrition_report(ingredients_section):
    prompt = f"Analyze these ingredients for Indian nutrition profile:\n{ingredients_section}"
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# (Rest of your UI code like extract_ingredients_section, sidebar, and buttons remain largely the same)
# ... [Extract/Clean functions go here] ...

# --- SESSION STATE ---
if "generated_recipe" not in st.session_state:
    st.session_state.generated_recipe = None

# --- UI LOGIC ---
with st.sidebar:
    st.title("Chef's Controls")
    meal_type = st.selectbox("Meal Type", ["Dinner", "Lunch", "Breakfast", "Snack", "Dessert"])
    creativity = st.slider("Innovation", 0, 10, 5)

st.markdown('<p class="main-header">👨‍🍳 AI Recipe Chef</p>', unsafe_allow_html=True)
ingredients_input = st.text_area("What's in your fridge?")

if st.button("🍳 Generate My Recipe", type="primary"):
    if ingredients_input:
        context_text, recipe_names = retrieve_recipes(ingredients_input)
        recipe_md = generate_recipe(ingredients_input, [], meal_type, creativity, context_text)
        st.session_state.generated_recipe = recipe_md
        st.session_state.retrieved_names = recipe_names

if st.session_state.generated_recipe:
    st.markdown(st.session_state.generated_recipe)