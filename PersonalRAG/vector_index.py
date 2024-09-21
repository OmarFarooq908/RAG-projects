import pinecone
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# Create index
index_name = "rag-chatbot-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric='cosine')
