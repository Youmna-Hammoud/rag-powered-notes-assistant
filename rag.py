# %% [markdown]
# # RAG-powered Assistant

# %% [markdown]
# ## RAG-Powered Assistant for PDF or Text Search

# %% [markdown]
# ### Project Goal:
# 
# Build a small app that answers questions from a document or a knowledge base using RAG (Retrieval-Augmented Generation).
# The pipeline: embed → store → retrieve → generate.

# %% [markdown]
# ### imports

# %%
from textwrap import wrap
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import fitz
from huggingface_hub import InferenceClient
import openai

# %% [markdown]
# ## Load Environment Variables

# %%
from dotenv import load_dotenv
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# %% [markdown]
# ## Load PDF File

# %%
# the pdf (math chapter)
data_folder = "./data/"

# %%
# Load the PDF file

full_text = ""
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_folder, filename)
        doc = fitz.open(filepath)
        for page in doc:
            full_text += page.get_text()

# %% [markdown]
# ## Split Text into Chunks

# %%
chunk_size = 500 
chunks = wrap(full_text, chunk_size)

# %% [markdown]
# ## Embeddings

# %%
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = [embedding_model.encode(chunk) for chunk in chunks]
embeddings = np.array(embeddings, dtype="float32")


# %% [markdown]
# ## Store Embeddings in FAISS

# %%
import faiss

embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)


# %% [markdown]
# ## Retrieval function

# %%
def search(query, top_k=3):
    query_emb = embedding_model.encode(query)
    query_emb = np.array([query_emb], dtype="float32")
    distances, indices = index.search(query_emb, top_k)
    return [(chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]

# %%
client = InferenceClient(api_key=hf_api_key)

def answer_question(query):
    # Retrieve context
    relevant_chunks = search(query, top_k=3)
    context = "\n".join([chunk for chunk, _ in relevant_chunks])

    prompt = f"""Use the following pieces of context to answer the question at the end. Please follow the following rules:
                1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
                2. If you find the answer, write the answer in a concise way with five sentences maximum.
                3. Use ONLY the following context to answer the question. If the answer is not contained in the context, respond with "I don't know."         

                {context}

                Question: {query}

                Helpful Answer:"""
    messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{query}"}
        ]

    # Call Hugging Face Inference API
    response = client.chat.completions.create(
        messages=messages,
        model="Qwen/Qwen3-4B-Instruct-2507",
        max_tokens=300,
        temperature=0.3
    )

    return response.choices[0].message.content
