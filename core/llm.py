from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

def chat(messages):

    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content
