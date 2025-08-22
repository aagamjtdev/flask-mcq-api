import os
import uuid
import json
import random
from base64 import b64encode
from collections import OrderedDict

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -----------------------------
# ChromaDB setup (local folder)
# -----------------------------
# Note: this stores data under ./chromadb_data in the container.
# On Render this storage will be ephemeral across new deploys unless you use a persistent disk.
client = chromadb.PersistentClient(
    path="./chromadb_data",
    settings=Settings()
)

collection_name = "mcq_collection"
existing = [c.name for c in client.list_collections()]
if collection_name in existing:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# -----------------------------
# Embedding model (loads at import)
# -----------------------------
# Model download can be large. If this causes memory/build issues, remove for initial deploy.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def ordered_mcq(mcq):
    return OrderedDict([
        ("question", mcq.get("question") or ""),
        ("image", mcq.get("image") or None),
        ("options", mcq.get("options") or {}),
        ("answer", mcq.get("answer") or "")
    ])


def store_mcqs(user_id, title, description, mcqs, pdf_file):
    generatedQAId = str(uuid.uuid4())
    user_id_clean = str(user_id).strip().lower()

    mcqs_ordered = [ordered_mcq(mcq) for mcq in mcqs]
    mcqs_json = json.dumps(mcqs_ordered, ensure_ascii=False)

    text_for_embedding = f"{title} {description}"
    embeddings = embedding_model.encode([text_for_embedding]).tolist()

    metadata = {
        "user_id": user_id_clean,
        "title": title,
        "generatedQAId": generatedQAId,
        "description": description,
        "file_name": pdf_file,
        "mcqs": mcqs_json
    }

    collection.add(
        ids=[generatedQAId],
        documents=[user_id_clean],
        embeddings=embeddings,
        metadatas=[metadata]
    )

    return generatedQAId


def fetch_mcqs(user_id: str = None, generatedQAId: str = None):
    if user_id:
        user_id_clean = str(user_id).strip().lower()
        results = collection.get(where={"user_id": user_id_clean})
        if not results or not results.get("ids"):
            return []

        data = []
        for idx, meta in enumerate(results["metadatas"]):
            mcq_list = meta.get("mcqs", "[]")
            if isinstance(mcq_list, str):
                try:
                    mcq_list = json.loads(mcq_list)
                except Exception:
                    mcq_list = []

            mcqs_ordered = [ordered_mcq(mcq) for mcq in mcq_list]

            data.append({
                "id": results["ids"][idx],
                "document": results["documents"][idx] if "documents" in results else None,
                "metadata": {
                    "user_id": meta.get("user_id"),
                    "title": meta.get("title"),
                    "description": meta.get("description"),
                    "file_name": meta.get("file_name"),
                    "generatedQAId": meta.get("generatedQAId"),
                    "mcqs": mcqs_ordered
                }
            })
        return data

    # if no filters provided, return all
    results = collection.get(include=["documents", "metadatas"])
    data = []
    for i in range(len(results["ids"])):
        meta = results["metadatas"][i].copy()

        mcq_list = []
        if "mcqs" in meta:
            if isinstance(meta["mcqs"], str):
                try:
                    mcq_list = json.loads(meta["mcqs"])
                except json.JSONDecodeError:
                    mcq_list = []
            else:
                mcq_list = meta["mcqs"]

        mcq_list = [ordered_mcq(mcq) for mcq in mcq_list]

        data.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": {
                "user_id": meta.get("user_id"),
                "title": meta.get("title"),
                "description": meta.get("description"),
                "file_name": meta.get("file_name"),
                "generatedQAId": meta.get("generatedQAId"),
                "mcqs": mcq_list
            }
        })
    return data


def fetch_random_mcqs(generatedQAId: str, num_questions: int = None):
    results = collection.get(ids=[generatedQAId], include=["metadatas", "documents"])
    if not results or not results.get("ids"):
        return {"mcqs": []}

    meta = results["metadatas"][0]
    mcqs_list = json.loads(meta.get("mcqs", "[]"))

    if not mcqs_list:
        return {"mcqs": []}

    if num_questions and num_questions < len(mcqs_list):
        mcqs_list = random.sample(mcqs_list, num_questions)

    formatted_mcqs = []
    for mcq in mcqs_list:
        question = mcq.get("question", "")
        image = mcq.get("image", None)
        options = mcq.get("options", {})
        correct_label = mcq.get("answer")
        correct_text = options.get(correct_label)

        option_items = list(options.items())
        random.shuffle(option_items)

        labels = ["A", "B", "C", "D"]
        new_options = OrderedDict()
        new_answer = None
        for idx, (_, text) in enumerate(option_items):
            label = labels[idx]
            new_options[label] = text
            if text == correct_text:
                new_answer = label

        formatted_mcq = OrderedDict([
            ("question", question),
            ("image", image),
            ("options", new_options),
            ("answer", new_answer)
        ])

        formatted_mcqs.append(formatted_mcq)

    return json.dumps({"mcqs": formatted_mcqs}, ensure_ascii=False, indent=2)
