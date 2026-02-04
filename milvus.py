from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama
import os
import subprocess
import threading
import numpy as np
import json
import chromadb
from chromadb.config import Settings

from typing import List, Dict, Tuple

import warnings
warnings.filterwarnings("ignore")

# torch.backends.cudnn.enabled = False


import fitz
from tqdm import tqdm
import unicodedata
import re

import spacy
from spacy.language import Language
from spacy.lang.en import English

from typing import List, Dict, Tuple


ollama = Ollama()




# def run_ollama():
#     # set environment variable to suppress logs and redirect output
#     env = os.environ.copy()
#     env["OLLAMA_LOG_LEVEL"] = "error"
    
#     # run with suppressed output
#     subprocess.run(
#         ["ollama", "serve"],
#         env=env,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL
#     )

# thread = threading.Thread(target=run_ollama)
# thread.start()

# device = "cuda:0" if torch.cuda.is_available() else "cpu"





# if device == "cpu":
#     llm_model_name = "gemma3:270m"
# else:
#     llm_model_name = "gemma3:1b"

# print("LLM Name:", llm_model_name)

# prompt = "How to train a dog?"

# print(f"User's Prompt:\n{prompt}")
# print("-" * 100)
# print("Bot:")

# # stream the response
# stream = ollama.generate(
#     model=llm_model_name,
#     prompt=prompt,
#     stream=True,
#     options={
#         "num_predict": 256,
#         "temperature": 0.7
#     }
# )

# full_response = ""
# for chunk in stream:
#     chunk_text = chunk["response"]
#     print(chunk_text, end="", flush=True)
#     full_response += chunk_text

# print()


def extract_text_from_pdf(pdf_path: str, text_cleaning: bool) -> str:
    book = fitz.open(pdf_path)

    full_text = ""
    for page in tqdm(book):
        if text_cleaning:
            full_text += clean_text(page.get_text())
        else:
            full_text += (page.get_text())

    return full_text


def clean_text(raw_text: str) -> str:
    """
    cleans text by removing control chars, ligatures, and spacing artifacts.
    
    args:
        raw_text: Raw text extracted from PDF with formatting artifacts
        
    returns:
        Cleaned text suitable for NLP processing and chunking
        
    example:
        input: "Hello – world…   See  you\nsoon!"
        Output: "Hello - world... See you soon!"
    """

    if not raw_text:
        return ""

    # Normalize unicode (decompose ligatures, accents)
    # Example: "ﬁ" (ligature) → "fi", "café" → "cafe"
    text = unicodedata.normalize("NFKC", raw_text)

    # Remove control characters and other invisible chars (ASCII < 32 except tab/lf/cr)
    # Example: "Hello\x01world" → "Hello world"
    # Control characters are invisible codes for hardware control, like bells (),
    # escape sequences, or data transmission signals. Example: "Hello\x07" makes a beep sound.
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]", " ", text)

    # Replace various kinds of dashes and hyphens with standard hyphen
    # Example: "Price – $100" → "Price - $100"
    text = text.replace("\xad", "")  # soft hyphen
    text = text.replace("–", "-").replace("—", "-")

    # Remove zero-width and non-breaking spaces
    # Example: "Hello​world" → "Hello world"
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("\u2003", " ")

    # Collapse multiple spaces and tabs into single space
    # Example: "Hello    world" → "Hello world"
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse broken words caused by PDF line breaks
    # Example: "for-\nmation" → "formation"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Replace remaining newlines with spaces
    # Example: "Hello\nworld" → "Hello world"
    text = text.replace("\n", " ")

    # Fix spacing before punctuation
    # Example: "Hello , world !" → "Hello, world!"
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Remove multiple consecutive spaces again
    # Example: "Hello  world" → "Hello world"
    text = re.sub(r"\s{2,}", " ", text)

    # Strip leading/trailing whitespace
    # Example: "  Hello world  " → "Hello world"
    text = text.strip()

    return text



pdf_path = "/home/parsa/Desktop/chatbot_project/linear-algebra_theory_intuition_code-by-mike-x-cohen.pdf" 


# full_text = extract_text_from_pdf(pdf_path, False)
full_text = extract_text_from_pdf(pdf_path, True)
print(full_text[:10000])




def initialize_spacy_pipeline() -> Language:
    """
    initializes the spaCy's "sentencizer" pipeline.
    
    output:
        Language: spaCy's language pipeline with sentencizer.
    """

    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp


def process_text_chunks(sentences: List[str], chunk_size: int, overlap_size: int) -> List[List[str]]:
    """
    takes a list of sentences as strings and splits them into chunks with specified overlap.
    Ex. chunk_size = 8
        overlap_size = 2
        setences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        returns -> [[0, 1, 2, 3, 4, 5, 6, 7], [6, 7, 8, 9, 10, 11, 12, 13], [12, 13, 14]]
        
    inputs:
        sentences: list of sentences to be chunked.
        chunk_size: number of sentences per chunk.
        overlap_size: number of overlapping sentences between chunks.
        
    output:
        a list of sentence chunks.
    """

    if chunk_size <= overlap_size:
        raise ValueError("[ERROR] chunk_size must be greater than overlap_size")
        
    if not sentences:
        return []
        
    step = chunk_size - overlap_size
    return [sentences[i:i+chunk_size]
            for i in range(0, len(sentences)-overlap_size, step)
            ]

nlp = initialize_spacy_pipeline()

doc = nlp(full_text)
sentences = [str(sentence).strip() for sentence in doc.sents]

print("Some Examples:")
print(sentences[150])
print(sentences[180])


chunk_size = 12
overlap_size = 2

chunks = process_text_chunks(sentences, chunk_size, overlap_size)

print("Some Examples:")
print(chunks[20])
print(chunks[21])



chunked_text = []
for chunk in chunks:
    paragraph = " ".join(chunk).strip()
    chunked_text.append(paragraph)

print("Some Examples:")
print(chunked_text[20])
print("-"*100)
print(chunked_text[21])








MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "test"
DIM = 512
embedding_model_name = "all-minilm:22m"
top_k = 5


connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)



fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]

schema = CollectionSchema(fields, description="Text embeddings collection")



if COLLECTION_NAME not in [c.name for c in Collection.list()]:
    collection = Collection(name=COLLECTION_NAME, schema=schema)
else:
    collection = Collection(name=COLLECTION_NAME)


#-----------------------HNSW indexing algorithm with Cosine metric type-----------------
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}

#-----------------------HNSW indexing algorithm with Euclidean Distance metric type-----------------
# index_params = {
#     "index_type": "HNSW",
#     "metric_type": "L2",
#     "params": {"M": 16, "efConstruction": 200}
# }

#-----------------------HNSW indexing algorithm with inner product (dot product) metric type-----------------
# index_params = {
#     "index_type": "HNSW",
#     "metric_type": "IP",
#     "params": {"M": 16, "efConstruction": 200}
# }


#-----------------------IVF indexing algorithm-----------------
# index_params = {
#     "index_type": "IVF_FLAT",
#     "metric_type": "COSINE",  
#     "params": {"nlist": 128}  
# }


#-----------------------PQ indexing algorithm-----------------
# index_params = {
#     "index_type": "IVF_PQ",
#     "metric_type": "COSINE",
#     "params": {
#         "nlist": 128, 
#         "m": 16,       
#         "nbits": 8    
#     }
# }

collection.create_index(field_name="embedding", index_params=index_params)



chunked_text = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England."
]

ids = [f"id_{i}" for i in range(len(chunked_text))]



for i, text in enumerate(tqdm(chunked_text, desc="Storing embeddings")):
    embedding = ollama.embeddings(model=embedding_model_name, prompt=text)["embedding"]
    # pad یا truncate
    if len(embedding) < DIM:
        embedding += [0.0] * (DIM - len(embedding))
    elif len(embedding) > DIM:
        embedding = embedding[:DIM]

    collection.insert([{
        "id": ids[i],
        "embedding": embedding,
        "text": text
    }])




# users_prompt = "What is the determinant?"
# query_embedding = ollama.embeddings(model=embedding_model_name, prompt=users_prompt)["embedding"]

# # pad یا truncate
# if len(query_embedding) < DIM:
#     query_embedding += [0.0] * (DIM - len(query_embedding))
# elif len(query_embedding) > DIM:
#     query_embedding = query_embedding[:DIM]

# search_params = {"metric_type": "COSINE", "params": {"ef": 200}}
# results = collection.search(
#     data=[query_embedding],
#     anns_field="embedding",
#     param=search_params,
#     limit=top_k,
#     output_fields=["text"]
# )




# for i, item in enumerate(results[0]):
#     similarity = 1 - item.distance
#     print(f"Text {i+1}, Similarity Score: {round(similarity*100,3)}%")
#     print(item.entity.get("text"))
#     print("-"*100)



# ------------------ Utility Functions ------------------
def embed_text(text: str, model_name: str):
    response = ollama.embeddings(model=model_name, prompt=text)
    return response["embedding"]

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ------------------ User Query ------------------
users_prompt = "What is the determinant?"
top_k = 5
DIM = 384

query_embedding = embed_text(users_prompt, embedding_model_name)

if len(query_embedding) < DIM:
    query_embedding += [0.0] * (DIM - len(query_embedding))
elif len(query_embedding) > DIM:
    query_embedding = query_embedding[:DIM]

# ------------------ Milvus Search ------------------
search_params = {"metric_type": "COSINE", "params": {"ef": 200}}

results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=top_k,
    output_fields=["text"]
)

relevant_chunks = []
for i, item in enumerate(results[0]):
    similarity = 1 - item.distance
    relevant_chunks.append({
        "text": item.entity.get("text"),
        "embedding": item.entity.get("embedding"),
        "similarity_score": similarity
    })

# ------------------ Display Retrieved Context ------------------
print("\nRetrieved Context:\n")
for i, chunk in enumerate(relevant_chunks):
    print(f"Text {i+1}, Similarity Score: {round(chunk['similarity_score']*100,3)}%")
    print(chunk["text"])
    print("-"*100)

# ------------------ (Mock) Generated Answer ------------------
generated_answer = """
The determinant of a matrix is calculated by multiplying the diagonals 
and subtracting the products of the off-diagonals for a 2x2 matrix.
For larger matrices, methods like cofactor expansion or row reduction can be used.
"""

# ------------------ Auto-generate Questions for Evaluation ------------------
def generate_questions_from_answer(answer_text, n_questions=3):
    prompt = f"Generate exactly {n_questions} clear questions from the following answer. Return as a JSON list only.\n\nAnswer:\n{answer_text}"
    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        questions = json.loads(response.text)
        if not isinstance(questions, list):
            raise ValueError
        return questions
    except:
        return [
            "How do you calculate the determinant of a matrix?",
            "What is the formula for a 2x2 matrix determinant?",
            "What methods exist to compute determinants for larger matrices?"
        ]

generated_questions = generate_questions_from_answer(generated_answer, n_questions=3)

# ------------------ Evaluate Answer Relevance ------------------
user_prompt_embedding = embed_text(users_prompt, embedding_model_name)
similarities = {}

for question in generated_questions:
    question_embedding = embed_text(question, embedding_model_name)
    similarity = cosine_similarity(user_prompt_embedding, question_embedding)
    similarities[question] = similarity

# ------------------ Report ------------------
print("\nAnswer Relevance Evaluation:\n")
print(f"User Prompt:\n{users_prompt}")
print("-"*100)
for q, score in similarities.items():
    print(f"Generated Question: {q}")
    print(f"Similarity Score: {round(score,3)}")
    print("-"*60)

answer_relevance_score = sum(similarities.values()) / len(similarities)
print(f"\nFinal Answer Relevance Score: {round(answer_relevance_score,3)}")
print("="*100)
