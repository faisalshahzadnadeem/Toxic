import os
from dotenv import load_dotenv

load_dotenv()

# We ask for the name of the variable here, not the key itself!
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")