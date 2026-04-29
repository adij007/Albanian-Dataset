import duckdb
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# ==========================================
# 1. Configuration
# ==========================================
INPUT_CSV = 'albanian_dictionary_dataset.csv'
DB_FILE = 'albanian_vectors.db'
# This model is excellent for Albanian and is lightweight (perfect for high-school/student projects)
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' 

def main():
    print(f"--- Loading Model: {MODEL_NAME} ---")
    model = SentenceTransformer(MODEL_NAME)

    # 2. Load Data
    print("Reading CSV...")
    # We only need cleaned_text for embeddings, but we keep filename for metadata
    df = pd.read_csv(INPUT_CSV)
    
    # Handle any empty rows to prevent model errors
    df['cleaned_text'] = df['cleaned_text'].fillna("").astype(str)
    
    # 3. Generate Embeddings
    print(f"Generating embeddings for {len(df)} entries... (This uses CPU/GPU)")
    # We use 'convert_to_numpy' because DuckDB handles numpy arrays well
    embeddings = model.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
    
    # Convert embeddings to a list of lists so pandas can insert them as a column
    df['embedding'] = embeddings.tolist()

    # 4. Initialize DuckDB and Store Data
    print(f"Connecting to DuckDB: {DB_FILE}...")
    con = duckdb.connect(DB_FILE)

    # Install and load the VSS extension for future similarity searches
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")

    # Create the table
    # We store the embedding as a FLOAT[384] (the dimension of MiniLM-L12)
    con.execute("""
        CREATE OR REPLACE TABLE dictionary_embeddings (
            filename VARCHAR,
            cleaned_text TEXT,
            timestamp VARCHAR,
            embedding FLOAT[384]
        )
    """)

    # Insert the dataframe into DuckDB
    print("Inserting data into DuckDB...")
    con.append('dictionary_embeddings', df[['filename', 'cleaned_text', 'timestamp', 'embedding']])

    # 5. Verify and Sample Search
    total_rows = con.execute("SELECT count(*) FROM dictionary_embeddings").fetchone()[0]
    print(f"Successfully stored {total_rows} vectors in DuckDB.")

    # Example: How to query a vector from the DB later
    sample = con.execute("SELECT cleaned_text, embedding FROM dictionary_embeddings LIMIT 1").fetchone()
    print("\n--- Sample Entry ---")
    print(f"Text: {sample[0][:50]}...")
    print(f"Vector (first 5 dimensions): {sample[1][:5]}")

    con.close()

if __name__ == "__main__":
    main()