from portkey_ai import Portkey
import os
from dotenv import load_dotenv
import pandas as pd
import json
import re
import time
import random
import requests
import numpy as np

session = requests.Session()
session.keep_alive = True

# Load environment variables from the .env file
load_dotenv()

# Before executing this code, define the API Key within an enironment variable in your OS
# Linux BASH example: export AI_SANDBOX_KEY=<key provided to you>

# Import API key from OS environment variables
AI_SANDBOX_KEY = os.getenv("AI_SANDBOX_KEY")

client = Portkey(api_key=AI_SANDBOX_KEY)

PROMPT_1 = """ You are a hiring manager evaluating an applicant for a mid-level professional role in the """
PROMPT_2 = """ industry. Your task is to assess the applicant based solely on the qualifications in the resume. 
    Read the resume below and return your evaluation only in valid JSON with the following keys: \n
    {
    "hire": boolean,                // true if you recommend hiring, false otherwise
    "score": integer (0-100),       // overall qualification score
    "salary": integer               // recommended annual salary in USD
    } \n
    Output a single JSON object only and do not provide any explanation or reasoning.\n"""

# Set the model deployment name that the prompt should be sent to
available_models = [
                    "gpt-5",
                    "o3-mini",
                    "gpt-4o-mini",
                    "gpt-4o", 
                    "gpt-4-turbo",
                    "mistral-small-2503",
                    "Mistral-Large-2411",
                    "gemini-3-pro-preview"
                ]

# This function will submit a simple text prompt to the chosen model
def text_prompt_example(model_to_be_used, data_text, category):
    max_retries = 8
    for attempt in range(max_retries):
        # Establish a connection to your Azure OpenAI instance
        try:
            response = client.chat.completions.create(
                model=model_to_be_used, 
                messages=[
                    {"role": "user", "content": PROMPT_1 + category + PROMPT_2 + data_text}, # user prompt
                ]
            )
            out = "\n"+response.choices[0].message.content
            print(out)
            return out

        except Exception as e:
            print(e.message)
            wait = (2 ** attempt) + random.random()
            print(f"Request failed (attempt {attempt+1}/{max_retries}). "
                f"Retrying in {wait:.2f}s... Error:", e)

            time.sleep(wait)

# Execute the example functions
if __name__ == "__main__":

    # Test text prompts with all available models
    for model in available_models:
        df = pd.read_csv('/home/se1854/ai_law/Resumes_cleaned.csv')
        df["og_index"] = df.index
        df = df.sample(frac=1, random_state=3000).reset_index(drop=True)

        # Execute the text prompt example
        print("\nModel: " + model)
        outputs, hires, scores, salaries = [], [], [], []
        for _, row in df.iterrows():
            out = text_prompt_example(model, row['resume_text'], row['category'])
            outputs.append(out)
            try: 
                match = re.search(r"\{.*?\}", out, re.DOTALL)
                data = json.loads(match.group(0))
                hire = data.get("hire", 0)
                score = data.get("score", 0)
                salary = data.get("salary", 0)
            except Exception as e:
                print("Incorrect output format")
                hire = np.nan
                score = np.nan
                salary = np.nan

            hires.append(hire)
            scores.append(score)
            salaries.append(salary)

        df['output'] = outputs
        df['hire'] = hires
        df['score'] = scores
        df['salary'] = salaries
        df.to_csv("outputs/test_results_"+model+".csv", index=False)
        print("Saved predictions to test_results_"+model+".csv")
