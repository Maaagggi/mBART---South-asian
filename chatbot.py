import os
from dotenv import load_dotenv
import openai
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

# Load environment variables from .env file
load_dotenv()

# Access API key from .env
openai.api_key = os.getenv('OPENAI_API_KEY')


def get_gpt_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Replace with the desired GPT-3.5 engine
        prompt=prompt,
        max_tokens=1024,  # Adjust the maximum number of tokens generated
        n=1,  # Number of completions to generate (set to 1 for single response)
        stop=None,  # Optional stop sequence to terminate generation
        temperature=0.7,  # Adjust the randomness of the generated text (0 for deterministic)
    )

    return response.choices[0].text.strip()  # Extract the generated text


def generate_response(user_input):
    prompt = f"The user asks: '{user_input}'"
    response = get_gpt_response(prompt)
    return response
