import fitz  # PyMuPDF
from google import genai
from PIL import Image
import json
import io
import os
import asyncio
import re
from tenacity import retry, stop_after_attempt, wait_exponential

client = genai.Client()
# ---------------------------
# CONFIG
# ---------------------------

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
PDF_PATH = r"BT R24 Regulations NIT ANP along with Annexure.pdf"

# ---------------------------
# HELPERS
# ---------------------------

def extract_json(text):
    if not text:
        return None

    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except:
        return None

# ---------------------------
# PROMPT
# ---------------------------

PROMPT = """
You are an information extraction system.

From the research paper page image extract:

1. Entities
2. Relationships between entities

Previously extracted entities from this paper (use these exactly as written for consistency if referring to the same concept):
{previous_entities}

Ensure entites are not isolated but are connected to each other thorugh relationships

Capture the formulas and equations if any

Ignore the references and bibliography parts, only extarct things relevant to science.

Return output in JSON format:

{{
  "entities": [
    {{"name": "", "type": ""}}
  ],
  "relationships": [
    {{"source": "", "relation": "", "target": ""}}
  ]
}}
"""

# ---------------------------
# PDF → Images
# ---------------------------

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)

    images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        pix = page.get_pixmap(dpi=300)

        img = Image.open(io.BytesIO(pix.tobytes("png")))

        images.append(img)

    return images


# ---------------------------
# IMAGE → LLM
# ---------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def extract_from_image(image, page_number, previous_entities):

    print(f"Processing page {page_number}...")
    
    # Format the prompt with previous entities
    # If no previous entities, pass "None"
    formatted_prompt = PROMPT.replace(
        "{previous_entities}",
        json.dumps(previous_entities, indent=2) if previous_entities else "None"
    )

    # Try using the new async client if using newer google-genai version
    # otherwise fallback to the user's original method call
    if hasattr(client, 'aio'):
        response = await client.aio.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[
                formatted_prompt,
                image
            ]
        )
    else:
        response = await client.models.generate_content_async(
            model='gemini-3-pro-preview',
            contents=[
                formatted_prompt,
                image
            ]
        )

    text = response.text
    
    # Extract token usage
    tokens = {
        "prompt_tokens": 0,
        "candidates_tokens": 0,
        "total_tokens": 0
    }
    
    if hasattr(response, 'usage_metadata') and response.usage_metadata is not None:
         tokens = {
            "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
            "candidates_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
            "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
         }

    try:
        data = extract_json(text)
        if not data:
             data = {"raw_output": text, "entities": []}

        print(data['entities'])
    except:
        data = {"raw_output": text, "entities": []}

    print(f"Finished page {page_number}")
    return {
        "page": page_number,
        "data": data,
        "tokens": tokens
    }


# ---------------------------
# MAIN PIPELINE
# ---------------------------

async def process_pdf(pdf_path):

    images = pdf_to_images(pdf_path)

    results = []
    
    # Keep track of all entities extracted so far to pass to the next page
    accumulated_entities = []
    
    total_prompt_tokens = 0
    total_candidates_tokens = 0

    for i, img in enumerate(images):
        
        # Sequentially await each page extraction
        result = await extract_from_image(img, i+1, accumulated_entities)
        
        # Track token usage
        total_prompt_tokens += result["tokens"]["prompt_tokens"]
        total_candidates_tokens += result["tokens"]["candidates_tokens"]
        
        # Add newly discovered entities to our running list
        if "entities" in result["data"]:
             for entity in result["data"]["entities"]:
                 # Simple deduplication by name
                 if not any(e.get("name") == entity.get("name") for e in accumulated_entities):
                     accumulated_entities.append(entity)

        results.append(result)

    print("\n--- TOKEN USAGE ---")
    print(f"Prompt Tokens: {total_prompt_tokens}")
    print(f"Completion Tokens: {total_candidates_tokens}")
    print(f"Total Tokens: {total_prompt_tokens + total_candidates_tokens}\n")

    return results


if __name__ == "__main__":

    results = asyncio.run(process_pdf(PDF_PATH))

    with open("kg_extraction.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Extraction complete!")
