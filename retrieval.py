from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from langchain_chroma import Chroma  # Updated import


"""
This script handles the retrieval of relevant documents from a Chroma vector database.
It uses HuggingFace embeddings to convert queries into vectors and searches the 
database for semantically similar content chunks.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the embedding model.
# We use the same model as in ingestion to ensure vector compatibility.
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
openai = OpenAI() # Uses OPENAI_API_KEY from environment

# -------- CONFIG --------
DB_PATH = Path("./chroma_db")

# Initialize the Chroma vector store.
# This connects to the persistent database created by the ingestion script.
# - persist_directory: Path to the stored database.
# - embedding_function: Function to convert text to vectors.
# - collection_metadata: Configuration for the vector space (cosine similarity).


db = Chroma(
    persist_directory=str(DB_PATH),
    embedding_function=embedding,
    collection_metadata={"hnsw:space": "cosine"},
)


# Define the query to search for.
query = "What is the difference between encoder and decoder in transformer architecture?"

def reranker(question, retrieved_chunks):
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = """
You are a document re-ranker.

You are provided with:
- a question
- a list of relevant chunks of text from a knowledge base
- each chunk has an associated chunk_id

The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.

Your task:
Rank the provided chunks by relevance to the question, with the most relevant chunk first.

Response format (STRICT):
Return ONLY a comma-separated list of chunk_ids in ranked order.
Do NOT include explanations, text, labels, brackets, or spaces except commas.

Example:
3,5,1,2,4

Return all provided chunk_ids exactly once.
"""

    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(retrieved_chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # print("MESSAGES : ",messages,"\n\n\n")
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content.strip()

    return reply



    


def retrieve(query):
    # Create a retriever interface from the vector store.
    # search_kwargs={"k": 5} specifies that we want to retrieve the top 15 most similar documents.
    retriever = db.as_retriever(search_kwargs={"k": 15})
    # Execute the retrieval.
    # This converts the query to a vector and performs a similarity search.
    relevant_data = retriever.invoke(query)

    relevant_data_send = relevant_data.copy()  # Make a copy to send to the reranker
    query_to_send = query
    
    reranked_chunks = reranker(query_to_send, relevant_data_send)

    # Convert "3,4,1,2,15..." → [3,4,1,2,15,...]
    rank_indices = [int(i.strip()) for i in reranked_chunks.split(",")]

    # If indices are 1-based (most LLM rerankers are), convert to 0-based
    rank_indices = [i - 1 for i in rank_indices]
    # for i in rank_indices:
    #     print("data type of rank_indices is ",type(i))

    # Reorder chunks
    reranked_data = [relevant_data[i] for i in rank_indices]
    
    return reranked_data[0:6]



if __name__ == "__main__":
    main()
    