import os
import warnings
import openai
from dotenv import load_dotenv

from openai import ChatCompletion

# Load environment variables from .env file
load_dotenv()

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# Initialize OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

openai.api_key = api_key


def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3",  # Use the gpt-3 model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,  # Adjust the maximum number of tokens generated
        n=1,  # Number of completions to generate (set to 1 for single response)
        stop=None,  # Optional stop sequence to terminate generation
        temperature=0.7,  # Adjust the randomness of the generated text (0 for deterministic)
    )
    return response.choices[0].message["content"].strip()  # Extract the generated text



def generate_response(user_input):
    prompt = f"The user asks: '{user_input}'"
    response = get_gpt_response(prompt)
    return response
