from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
#index_name = "rag-chatbot-index"
index_name = "gpt-customizations"
index = pc.Index(index_name)

# Load embedding model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    name: str
    age: float
    gender: str
    medical_condition: str
    doctor: str
    hospital: str
    #score: float

@app.post("/query", response_model=list[QueryResponse])
def query_database(request: QueryRequest):
    # Encode the query
    query_vector = model.encode(request.query).tolist()
    
    # Query Pinecone index
    results = index.query(vector=[query_vector], top_k=request.top_k, include_metadata=True)

    # Parse the response
    response = [
        QueryResponse(
            name=result['metadata'].get('Name', 'No Name'),
            age=result['metadata'].get('Age', 'No Age'),
            gender=result['metadata'].get('Gender', 'No Gender'),
            medical_condition=result['metadata'].get('Medical Condition', 'No Condition'),
            doctor=result['metadata'].get('Doctor', 'No Doctor'),
            hospital=result['metadata'].get('Hospital', 'No Hospital')
        ) for result in results['matches']
    ]
    return response