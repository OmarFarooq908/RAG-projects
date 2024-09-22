import os
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if it doesn't exist
index_name = "gpt-customizations"
print(os.getcwd())
"""

if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
            )
        )
"""
# Connect to the index
index = pc.Index(index_name)

# Load the dataset
df = pd.read_csv('./data/healthcare_dataset.csv')

# Initialize the sentence transformer model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2') # 384 dimensions
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Function to create a combined text field for embedding
def create_combined_text(row):
    return f"Name: {row['Name']}, Age: {row['Age']}, Gender: {row['Gender']}, Blood Type: {row['Blood Type']}, " \
           f"Condition: {row['Medical Condition']}, Admission Date: {row['Date of Admission']}, " \
           f"Doctor: {row['Doctor']}, Hospital: {row['Hospital']}, Insurance: {row['Insurance Provider']}, " \
           f"Billing Amount: {row['Billing Amount']}, Room Number: {row['Room Number']}, " \
           f"Admission Type: {row['Admission Type']}, Discharge Date: {row['Discharge Date']}, " \
           f"Medication: {row['Medication']}, Test Results: {row['Test Results']}"

# Embed and upsert data
for idx, row in df.iterrows():
    # Create combined text for embedding
    combined_text = create_combined_text(row)
    
    # Encode content and get embedding
    embedding = model.encode(combined_text).tolist()
    
    # Upsert data to Pinecone
    index.upsert([(str(idx), embedding, {'Name': row['Name'], 'Age': row['Age'], 'Gender': row['Gender'], 
                                          'Medical Condition': row['Medical Condition'], 
                                          'Doctor': row['Doctor'], 'Hospital': row['Hospital']})])

print("Data ingestion completed successfully.")
