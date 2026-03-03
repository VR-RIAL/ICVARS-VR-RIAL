import os
import requests
import json

API_KEY = "API_KEY"

API_URL = "https://api.mistral.ai/v1/chat/completions"

# Read the prompt from a text file
with open("prompt.txt", "r", encoding="utf-8") as file:
    prompt = file.read()

for i in range(1,2): # initially designed to run for multiple inputs at a time                                                                        # modify this accordingly
    with open(f"Inputs/{i}/input.txt", "r", encoding="utf-8") as file:
        userQuery = file.read()

    finalPrompt = prompt + userQuery

    # Call the LLM API
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": finalPrompt}],
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        output_text = response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        output_text = f"Error: An API request error occurred: {e}"
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        output_text = f"Error: Could not parse API response: {e}"

    with open(f"Inputs/{i}/output.txt", "w", encoding="utf-8") as file:
        file.write(output_text)

    try:
        json_start = output_text.find('{')
        json_end = output_text.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            extracted_json_string = output_text[json_start : json_end + 1]
            json_content = json.loads(extracted_json_string)
            with open(f"Inputs/{i}/requirements.json", "w", encoding="utf-8") as req_file:
                json.dump(json_content, req_file, indent=2)
            print(f"\nSuccessfully updated requirements.json with content from output.txt.")
        else:
            print(f"\nWarning: Could not find valid JSON structure in output.txt. requirements.json was not updated.")
    except json.JSONDecodeError:
        print(f"\nWarning: Extracted content is not valid JSON. requirements.json was not updated.")
    except Exception as e:
        print(f"\nError updating requirements.json: {e}")

    print(output_text)

