from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ollama
import os
import subprocess
import threading

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


#---------------------------------chromaDB--------------------------------
def initialize_vector_db(collection_name="RAG"):
    """
    
    """

    db_client = chromadb.Client(Settings(
                        anonymized_telemetry=False, # disables sending anonymous usage data to ChromaDB
                        # persist_directory="vector_db" # optional: persist data to disk
                        ))
    
    collection = db_client.create_collection(
                                    name=collection_name,
                                    metadata={"hnsw:space":"cosine"} # similarity search method
                                )
    
    return collection



ids = [f"id_{i}" for i in range(len(chunked_text))]

embedding_model_name = "all-minilm:22m"
ctx_length = 512

# embedding_model_name = "jina/jina-embeddings-v2-base-en"
# ctx_length = 8192

collection = initialize_vector_db("test")

for i in tqdm(range(len(chunked_text)), desc="Storing Embeddings"):
    text = chunked_text[i]
    words_in_text = text.split(" ")
    num_words_in_text = words_in_text.__len__()

    if num_words_in_text > ctx_length:
        print("You should truncate the text.")
        # print(f"Len: {len(text)}, Num Words: {len(text.split(' '))}, Text:\n{text}")
        # break
        
    try:
        response = ollama.embeddings(model=embedding_model_name, prompt=text)
        collection.add(
                ids=ids[i],
                documents=text,
                embeddings=response["embedding"]
                )
    except:
        print(f"Len: {len(text)}, Num Words: {len(text.split(' '))}, Text:\n{text}")
    




users_prompt = "What is the determinant?"
# users_prompt = "What is hydraulic fracturing?"
# users_prompt = "How stress field can affect the direction of propagation of hydraulic fractures?"

top_k = 20
prompt_embedding = ollama.embeddings(model=embedding_model_name, prompt=users_prompt)["embedding"]

results = collection.query(
    query_embeddings=[prompt_embedding],
    n_results=top_k,
    include=["documents", "embeddings", "metadatas", "distances"],
    # where={"domain": {"$eq": domain}}
)

relevant_chunks = []
if not results["documents"] or not results["documents"][0]:
    # return relevant_chunks  # return an empty list if no results found
    print("No relevant results")

top_k = min(top_k, len(results["documents"][0]))
for i in range(top_k):
    relevant_chunks.append({
        "text": results["documents"][0][i],
        "embedding": results["embeddings"][0][i],
        "similarity_score": 1 - results["distances"][0][i]  # convert distance to similarity
    })




for i, chunk in enumerate(relevant_chunks):
    print(f"Text {i+1}, Similarity Score: {round(chunk['similarity_score']*100, 3)}%")
    print(chunk["text"])
    print("-"*100)





#--------------------testing streaming+ollama-----------------------------------
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
# print("Device:", device)
