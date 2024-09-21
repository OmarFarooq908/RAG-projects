import os
import pinecone
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

# Create index if it doesn't exist
index_name = "rag-chatbot-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric='cosine')

# Connect to the index
index = pinecone.Index(index_name)

# Load the dataset
df = pd.read_csv('data/healthcare_dataset.csv')

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed and upsert data
for idx, row in df.iterrows():
    # Encode content and get embedding
    embedding = model.encode(row['content']).tolist()
    
    # Upsert data to Pinecone
    index.upsert([(str(row['id']), embedding, {'title': row['title']})])

print("Data ingestion completed successfully.")