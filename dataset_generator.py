import google.generativeai as genai
import json
import time
import os
from tqdm import tqdm
from recipe_list import RECIPE_NAMES # Imports the list from recipe_list.py

# --- CONFIGURATION ---
# 1. PASTE YOUR GEMINI API KEY HERE
API_KEY = "AIzaSyAQrHoZnsvqdows75cRaqkFgGk0JQZ_fRs" # "Your_l0ng_API_key_goes_h3re"

# This is the prompt we will use to generate each recipe
GENERATION_PROMPT = """
You are a master Indian chef and recipe writer.
I need you to write a clear, high-quality recipe for: {recipe_name}

Your response MUST be in the following exact JSON format, and nothing else.
Do not add any text before or after the JSON block.

{{
  "title": "Your Recipe Title Here",
  "ingredients": "List all ingredients here as a single string, with each ingredient on a new line.",
  "instructions": "List all instructions here as a single string, with each step on a new line and numbered (e.g., '1. ...', '2. ...')."
}}
"""

OUTPUT_FILE = "recipes.json"
FAILED_FILE = "failed_recipes.txt"

# --- SETUP ---

if not API_KEY:
    print("Error: API_KEY is not set. Please open 'generate_dataset.py' and paste your Gemini API key on Line 7.")
    exit()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

# --- FUNCTIONS ---

def generate_recipe(recipe_name):
    """Calls the Gemini API to generate a single recipe in JSON format."""
    prompt = GENERATION_PROMPT.format(recipe_name=recipe_name)
    try:
        response = model.generate_content(prompt)
        
        # Clean the response to find the JSON block
        text = response.text
        if '```json' in text:
            text = text.split('```json\n', 1)[1].rsplit('```', 1)[0]
        
        # Parse the JSON
        data = json.loads(text)
        
        # Basic validation
        if "title" in data and "ingredients" in data and "instructions" in data:
            return data
        else:
            print(f"  [Warning] Missing keys in JSON for: {recipe_name}")
            return None
            
    except Exception as e:
        print(f"  [Error] Failed to generate or parse for: {recipe_name}. Error: {e}")
        return None

def main():
    print(f"Starting dataset generation for {len(RECIPE_NAMES)} recipes...")
    print(f"Generated recipes will be saved to: {OUTPUT_FILE}")
    print(f"Failed recipes will be logged in: {FAILED_FILE}")

    all_recipes = []
    failed_recipes = []

    # Use tqdm for a progress bar
    for recipe_name in tqdm(RECIPE_NAMES, desc="Generating Recipes"):
        recipe_data = generate_recipe(recipe_name)
        
        if recipe_data:
            all_recipes.append(recipe_data)
        else:
            failed_recipes.append(recipe_name)
            
        # Be polite to the API, wait a moment between calls
        time.sleep(1) 

    # --- SAVE RESULTS ---
    
    # Save successful recipes
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_recipes, f, indent=4, ensure_ascii=False)
        
    print(f"\n--- SUCCESS ---")
    print(f"Successfully generated and saved {len(all_recipes)} recipes to '{OUTPUT_FILE}'.")

    # Save failed recipes
    if failed_recipes:
        with open(FAILED_FILE, 'w', encoding='utf-8') as f:
            for name in failed_recipes:
                f.write(f"{name}\n")
        print(f"\n--- FAILED ---")
        print(f"{len(failed_recipes)} recipes failed to generate. See '{FAILED_FILE}' for a list.")

if __name__ == "__main__":
    main()