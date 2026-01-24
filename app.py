import streamlit as st
import google.genai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import json
import os
import random
import time
import markdown  # for Markdown -> HTML conversion

try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # This handles the local environment where you might not have pysqlite3 installed
    pass

# --- CONFIGURATION ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "indian_recipes"
RECIPE_DATA_FILE = "recipes.json"
GENERATION_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

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
    /* Main App Styling */
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: 800;
        margin-bottom: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 1.2rem;
        color: gray;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }

    /* Recipe / Shopping / Calories Card */
    .recipe-container {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin: 20px auto 40px auto;
        max-width: 1000px;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333333;
    }

    .recipe-container h1 {
        color: #c0392b !important;
        font-family: 'Georgia', serif !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        border-bottom: 3px solid #e74c3c !important;
        padding-bottom: 10px !important;
        margin-bottom: 25px !important;
        text-align: center !important;
    }

    .recipe-container h2 {
        color: #e67e22 !important;
        font-family: 'Georgia', serif !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 30px !important;
        border-bottom: 1px solid #eee !important;
        padding-bottom: 5px !important;
    }

    .recipe-container p,
    .recipe-container li {
        color: #333333 !important;
        line-height: 1.6 !important;
        font-size: 1.05rem !important;
    }

    .recipe-container strong {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }

    /* Make long content scrollable if needed */
    .recipe-container-long {
        max-height: 700px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- GEMINI API SETUP ---
API_KEY = st.secrets["GEMINI_API_KEY"]

try:
    if API_KEY:
        genai.Client(api_key=API_KEY)
    model = genai.GenerativeModel(GENERATION_MODEL_NAME)
except Exception as e:
    st.error(f"Error initializing Gemini. Check API Key. {e}")
    st.stop()

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are "AI Recipe Chef," an expert Michelin-star Indian chef. 
Your goal is to create a NEW, creative, and delicious recipe based on the user's ingredients.

RULES:
1. Use the "Context Recipes" for inspiration on spice blends and techniques, but DO NOT copy them.
2. Adhere strictly to the user's "Dietary Preferences" and "Meal Type".
3. The recipe must be coherent and chemically plausible.
4. FORMATTING:
   - Title MUST start with a single hash (# Title).
   - Use double hash (##) for "Ingredients" and "Instructions".
   - Do NOT use asterisks for the title.
"""

# --- CACHED RESOURCES ---
@st.cache_resource
def get_resources():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Auto-rebuild logic to prevent KeyErrors / schema mismatches
    try:
        col = client.get_collection(CHROMA_COLLECTION_NAME)
        if col.count() > 0:
            first_item = col.peek(1)
            if 'full_text' not in first_item['metadatas'][0]:
                raise ValueError("Schema mismatch")
    except Exception:
        try:
            client.delete_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    if collection.count() == 0:
        populate_database(collection, embedding_model)

    return embedding_model, collection, client


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
            metadatas.append(
                {
                    "title": recipe.get('title', ''),
                    "full_text": f"Title: {recipe.get('title', '')}\n\nIngredients:\n{recipe.get('ingredients', '')}\n\nInstructions:\n{recipe.get('instructions', '')}",
                }
            )
            ids.append(str(i))

        embeddings = model.encode(documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as e:
        st.error(f"Error populating database: {e}")
        st.stop()


# --- CORE HELPERS ---
def retrieve_recipes(query, k=3):
    embedding_model, collection, _ = get_resources()
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding, n_results=k, include=["metadatas"]
    )

    context_text = ""
    recipe_names = []

    if results and results['metadatas']:
        for i, meta in enumerate(results['metadatas'][0]):
            context_text += (
                f"--- Context {i+1}: {meta['title']} ---\n{meta['full_text']}\n\n"
            )
            recipe_names.append(meta['title'])

    return context_text, recipe_names


def generate_recipe(ingredients, preferences, meal_type, creativity, context):
    prompt = f"""
{SYSTEM_PROMPT}

USER REQUEST:
- Ingredients: {ingredients}
- Dietary Preferences: {', '.join(preferences) if preferences else 'None'}
- Meal Type: {meal_type}
- Creativity Level: {creativity}/10

CONTEXT RECIPES (For Inspiration):
{context}

Create the recipe now:
"""

    temp = 0.2 + (creativity * 0.08)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=temp),
        )
        if not response or not hasattr(response, "text"):
            return "Recipe generation failed. Please try again."
        return response.text
    except Exception as e:
        return f"Error generating recipe: {e}"


def extract_ingredients_section(markdown_text: str) -> str:
    """Extract the 'Ingredients' section from the generated recipe markdown."""
    lower = markdown_text.lower()
    if "## ingredients" not in lower:
        return ""

    start = lower.index("## ingredients")
    # slice original text from that position to preserve case
    sliced = markdown_text[start:]
    parts = sliced.split("\n## ")
    section = parts[0]  # up to next heading
    lines = section.splitlines()
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[1:]).strip()


def clean_shopping_list(text: str) -> str:
    """Try to ensure ingredients appear as bullet points."""
    if not text:
        return ""

    # If already has '-' bullets, leave mostly as-is
    if "- " in text:
        return text.strip()

    # Convert common '* ingredient' patterns to '- ingredient'
    formatted = text.replace(" * ", "\n- ").replace("* ", "- ")
    return formatted.strip()


def generate_nutrition_report(ingredients_section):
    prompt = f"""
You are “AI Nutritionist”, a certified dietitian specialized in Indian and global cuisines.

Your goal is to analyze the ingredient list of a recipe and provide a detailed nutritional profile.

Follow these rules strictly:

1. Assume standard ingredient quantities if not given, based on typical Indian home-cooking portions.
2. Output the full nutrition breakdown as:
   - Total Calories (kcal)
   - Calories per Serving
   - Protein (g)
   - Carbohydrates (g)
   - Dietary Fiber (g)
   - Total Fat (g)
   - Saturated Fat (g)
   - Cholesterol (mg)
   - Sodium (mg)
   - Sugar (g)

3. Also provide:
   - Recommended serving size
   - Whether the dish is high/low in calories, protein, carbs, fat, or sodium
   - A short 3–4 line “Health Insight”
   - Suggestions for making it healthier (if relevant)

4. Structure your output exactly like this:

# Nutrition Breakdown
**Total Calories:** ___ kcal  
**Calories per Serving:** ___ kcal  
**Protein:** ___ g  
**Carbohydrates:** ___ g  
**Fiber:** ___ g  
**Fat:** ___ g  
**Saturated Fat:** ___ g  
**Cholesterol:** ___ mg  
**Sodium:** ___ mg  
**Sugar:** ___ g  

# Health Insights
...

# Serving Suggestion
...

# Make It Healthier
- ...
- ...
- ...

INGREDIENTS:
{ingredients_section}
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating nutrition report: {e}"



# --- SESSION STATE INITIALIZATION ---
if "generated_recipe" not in st.session_state:
    st.session_state.generated_recipe = None
    st.session_state.used_ingredients = ""
    st.session_state.context_text = ""
    st.session_state.retrieved_names = []
    st.session_state.last_method = ""  # "custom" or "surprise"


def run_pipeline(ingredients_str: str, preferences, meal_type, creativity, method: str):
    """Runs retrieval + generation and stores results in session_state."""
    with st.spinner("👨‍🍳 Chef is cooking up your recipe..."):
        context_text, recipe_names = retrieve_recipes(ingredients_str)
        recipe_md = generate_recipe(
            ingredients_str, preferences, meal_type, creativity, context_text
        )

    st.session_state.generated_recipe = recipe_md
    st.session_state.used_ingredients = ingredients_str
    st.session_state.context_text = context_text
    st.session_state.retrieved_names = recipe_names
    st.session_state.last_method = method


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3565/3565418.png", width=100)
    st.title("Chef's Controls")

    st.subheader("⚙️ Configuration")
    meal_type = st.selectbox(
        "Meal Type", ["Dinner", "Lunch", "Breakfast", "Snack", "Dessert"]
    )

    st.subheader("🥦 Preferences")
    preferences = []
    if st.checkbox("Vegetarian"):
        preferences.append("Vegetarian")
    if st.checkbox("Vegan"):
        preferences.append("Vegan")
    if st.checkbox("Gluten-Free"):
        preferences.append("Gluten-Free")
    if st.checkbox("Spicy 🔥"):
        preferences.append("Spicy")
    if st.checkbox("Quick (< 30 mins)"):
        preferences.append("Quick and Easy")

    st.subheader("🎨 Creativity")
    creativity = st.slider(
        "Innovation Level",
        0,
        10,
        5,
        help="0 = Traditional, 10 = Wild Fusion",
    )

# --- MAIN HEADER ---
st.markdown('<p class="main-header">👨‍🍳 AI Recipe Chef</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Novel Culinary Creation via Retrieval-Augmented Generation</p>',
    unsafe_allow_html=True,
)

# --- INPUT AREA + BUTTONS ---
col1, col2 = st.columns([2, 1])

with col1:
    ingredients_input = st.text_area(
        "What's in your fridge?",
        height=100,
        placeholder="e.g., Paneer, Tomato, Cream, Kasuri Methi",
    )

with col2:
    st.write("")
    st.write("")
    surprise_clicked = st.button("🎲 Surprise Me!", use_container_width=True)

generate_clicked = st.button("🍳 Generate My Recipe", type="primary", use_container_width=True)

# --- HANDLE BUTTON ACTIONS ---

# Option B: Surprise Me directly generates a recipe from random ingredients
if surprise_clicked:
    try:
        data = load_recipe_data()
        random_recipe = random.choice(data)

        # Use first few ingredients lines as the base ingredient list
        raw_ings = random_recipe.get("ingredients", "").split("\n")
        raw_ings = [r.strip("-• ").strip() for r in raw_ings if r.strip()]
        base_ings = ", ".join(raw_ings[:5]) if raw_ings else random_recipe.get("title", "")

        if base_ings:
            run_pipeline(base_ings, preferences, meal_type, creativity, method="surprise")
        else:
            st.warning("Could not extract ingredients from the random recipe.")
    except Exception as e:
        st.error(f"Error reading recipes: {e}")

elif generate_clicked:
    if not ingredients_input.strip():
        st.warning("Please enter some ingredients first!")
    else:
        run_pipeline(ingredients_input.strip(), preferences, meal_type, creativity, method="custom")

# --- DISPLAY OUTPUT IF A RECIPE EXISTS ---
if st.session_state.generated_recipe:
    generated_recipe = st.session_state.generated_recipe
    used_ingredients = st.session_state.used_ingredients
    context_text = st.session_state.context_text
    recipe_names = st.session_state.retrieved_names
    last_method = st.session_state.last_method

    # Tabs: Recipe, Shopping List, Calories, How it Works
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📖 The Recipe", "🛒 Shopping List", "🔥 Calories", "🧐 How it Works"]
    )

    # ---------- TAB 1: THE RECIPE ----------
    with tab1:
        if last_method == "surprise":
            st.caption(f"🎲 Surprise Me used these base ingredients: {used_ingredients}")
        else:
            st.caption(f"Using ingredients: {used_ingredients}")

        # Download button at the TOP
        st.download_button(
            label="📥 Download Recipe",
            data=generated_recipe,
            file_name="my_ai_recipe.md",
            mime="text/markdown",
        )

        # Convert Markdown -> HTML and wrap in styled container
        html_content = markdown.markdown(
            generated_recipe, extensions=["extra", "fenced_code", "tables"]
        )

        final_html = f"""
        <div class="recipe-container recipe-container-long">
            {html_content}
        </div>
        """

        st.html(final_html)

    # ---------- TAB 2: SHOPPING LIST ----------
    with tab2:
        st.markdown('<h2 style="margin-bottom: 10px;">Quick Shopping List</h2>', unsafe_allow_html=True)

        ingredients_section = extract_ingredients_section(generated_recipe)
        if ingredients_section:
            cleaned_ingredients = clean_shopping_list(ingredients_section)
            shopping_html = markdown.markdown(
                cleaned_ingredients, extensions=["extra", "fenced_code", "tables"]
            )

            shopping_final_html = f"""
            <div class="recipe-container recipe-container-long">
                {shopping_html}
            </div>
            """
            st.html(shopping_final_html)
        else:
            st.write("Could not auto-extract ingredient list. Please refer to the full recipe.")

    # ---------- TAB 3: CALORIES ----------
    with tab3:
        st.header("🔥 Nutrition & Calories")

        ingredients_section = extract_ingredients_section(generated_recipe)

        if ingredients_section:
            nutrition_report = generate_nutrition_report(ingredients_section)

            html_nutrition = markdown.markdown(
                nutrition_report,
                extensions=["extra"]
            )

            final_nutrition_html = f"""
            <div class="recipe-container recipe-container-long">
                {html_nutrition}
            </div>
            """
            st.html(final_nutrition_html)
        else:
            st.write("Couldn't extract ingredients for nutrition analysis.")


    # ---------- TAB 4: HOW IT WORKS ----------
    with tab4:
        st.success("RAG Pipeline Successful!")
        st.write(f"**User / Surprise Ingredients:** {used_ingredients}")
        st.write(f"**Constraints:** {meal_type}, Preferences: {', '.join(preferences) if preferences else 'None'}, Creativity: {creativity}/10")
        st.write("---")
        st.subheader("Retrieved Context Recipes")
        for name in recipe_names:
            st.caption(f"✅ {name}")

        with st.expander("View Full Context Used"):
            st.text(context_text)

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built for SMST604 Research Project | Powered by Gemini & ChromaDB</div>",
    unsafe_allow_html=True,
)
