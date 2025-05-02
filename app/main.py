"""
main.py - catalogue chatbot is used to create a chatbot agent with OpenAI that help users.
"""
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List

import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CATALOGUE_API = os.getenv("CATALOGUE_API")

# === Shared memory (global) ===
INDEX_LOCK = threading.Lock()
INDEX = None
books = []
books_ids = {}


def embed_text(text: str) -> list:
    """
    This method embed a text.
    :param text: The text to embed.
    :return: the embedded text.
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def build_faiss_index_from_api(api_url: str):
    """
    Build the books index from the catalogue API books.
    :param api_url: The catalogue API URL.
    :return: Books index.
    """
    global INDEX, books, books_ids

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        books_raw = response.json()

        embeddings = []
        ids = []

        for book in books_raw:
            if not book.get("summary"):
                continue
            emb = embed_text(book["summary"])
            embeddings.append(emb)
            ids.append(int(book["isbn"]))

        embeddings_np = np.array(embeddings, dtype=np.float32)

        dim = embeddings_np.shape[1]
        idx = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        idx.add_with_ids(embeddings_np, np.array(ids, dtype=np.int64))

        with INDEX_LOCK:
            INDEX = idx
            books = books_raw
            books_ids = {int(l["isbn"]): l for l in books_raw if "isbn" in l}

    except Exception as e:
        print(f"Unable to update indexes : {e}")


def schedule_refresh(api_url: str, interval_seconds: int = 600):
    """
    Schedule a fresh of the INDEX.
    :param api_url:
    :param interval_seconds:
    :return:
    """

    def loop():
        while True:
            build_faiss_index_from_api(api_url)
            time.sleep(interval_seconds)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


def chatbot_conversation(messages: List[dict]) -> str:
    """
    Generate a response by a message history.
    :param messages: The message history.
    :return: A response.
    """
    if INDEX is None:
        raise RuntimeError("L'application est en cours d'initialisation...")

    last_question = messages[-1]["content"]

    vector = np.array([embed_text(last_question)], dtype=np.float32).reshape(1, -1)

    with INDEX_LOCK:
        _, ids = INDEX.search(x=vector, k=5)
        candidates = [books_ids[i] for i in ids[0] if i in books_ids]

    book_text = "\n".join(
        [
            f"- {book['title']}: {book['summary']}" for book in candidates
        ]
    )

    full_message = [{
        "role": "system",
        "content": (
                "Tu es un bibliothécaire nommé ChatXUM. " +
                "Tu peux discuter librement, répondre à des questions sur les livres, recommander des livres si nécessaires. " +
                "Si tu ne trouves pas un livre, ou si on te demande, tu rafraichis ta base de livres toutes les 10 minutes." +
                "Si on te demande ta technologie ou toute question relative à toi, tu es un bibliothécaire construit par Corentin Le Guen." +
                f"Ta base de livres, contient {len(books_ids)} mais elle peut évoluer." +
                "Voici ta base de livres:\n" +
                (book_text or "pas encore de livres disponibles.")
        )
    }] + messages

    response = client.chat.completions.create(
        model="gpt-4",
        messages=full_message,
    )

    return response.choices[0].message.content


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Run this script on app startup
    :return:
    """
    api_url = CATALOGUE_API
    build_faiss_index_from_api(api_url)
    schedule_refresh(api_url)
    yield


app = FastAPI(lifespan=lifespan)


class ChatHistory(BaseModel):
    """
    Chat history with the user.
    """
    messages: List[dict]


@app.get("/health")
def get_health():
    """
    Simple endpoint to see if the server is up.
    :return: up
    """
    return {"status": "up"}


@app.post("/ask")
def ask(history: ChatHistory):
    """
    Ask a question to the chatbot with history.
    :param history: The chat history
    :return: A response.
    """
    if INDEX is None:
        return {"message": "index not ready"}

    try:
        reply = chatbot_conversation(history.messages)
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}
