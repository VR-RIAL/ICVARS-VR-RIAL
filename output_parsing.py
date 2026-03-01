import requests
import base64
import json

API_KEY = "API_KEY_HERE"

IMAGE_PATH = "Inputs/{number}/Vastu_Layout_Output_v4.png"
OUTPUT_JSON_PATH = "Inputs/{number}/image_analysis.json"
MODEL_NAME = "mistral-large-latest"
API_URL = "https://api.mistral.ai/v1/chat/completions"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_floor_plan():

    if API_KEY == "YOUR_MISTRAL_API_KEY_HERE":
        print("Error: Please replace 'YOUR_MISTRAL_API_KEY_HERE' with your actual Mistral API key.")
        return

    print("Encoding image to base64...")
    base64_image = encode_image_to_base64(IMAGE_PATH)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    prompt = f"""Analyze the attached floor plan image. Identify all the rooms, their approximate dimensions if possible, and any special features. Please structure your response as a JSON object with the same format as the following example:

    Example format:
    ```json
    {{
      "house_dimensions": {{
        "width": 50,
        "length": 60,
        "unit": "feet"
      }},
      "rooms": {{
        "bedrooms": {{
          "count": 3,
          "features": ["attached bathroom", "balcony"]
        }},
        "living_room": {{
          "features": ["skylights", "high ceiling"]
        }},
        "kitchen": {{
          "min_area": 300,
          "unit": "square feet"
        }}
      }}
    }}
    ```

    Ensure that each room mentioned is in singular form. For example, mention a room as 'bedroom', not 'bedrooms'.
    Now, analyze the provided image and return only the JSON object.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "temperature": 0.1,
        "top_p": 1,
        "max_tokens": 2048,
        "stream": False,
        "safe_prompt": False,
        "random_seed": 1337,
    }

    print("Sending request to Mistral AI API...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        print("Processing response...")

        response_json = response.json()

        json_string = response_json['choices'][0]['message']['content'].strip()
        if json_string.startswith("```json"):
            json_string = json_string[7:-3].strip()

        final_json = json.loads(json_string)

        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(final_json, f, indent=2)

        print(f"Successfully created JSON analysis at: {OUTPUT_JSON_PATH}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with the API request: {e}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Could not parse the JSON from the API response: {e}")
        print("Raw response content:", response.text)

if __name__ == "__main__":
    analyze_floor_plan()