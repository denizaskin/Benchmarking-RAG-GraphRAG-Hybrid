import os
import sys
import asyncio
import csv
import re
import numpy as np
import ast
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pymupdf
import aiofiles
import igraph
import leidenalg
import torch
import httpx  # for catching ReadTimeout errors
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

# Import LLMs.
from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx

# Load environment variables.
load_dotenv()

# Set up WatsonX credentials.
url = os.getenv("WATSONX_URL")
apikey = os.getenv("API_KEY")
project_id = os.getenv("PROJECT_ID")
openai_apikey = os.getenv("API_KEY")  # or os.getenv("OPENAI_API_KEY")

# Define model IDs.
model_id_llama = "meta-llama/llama-3-405b-instruct"
model_id_mistral = "mistralai/mixtral-8x7b-instruct-v01"
model_id_code = "ibm/granite-34b-code-instruct"
model_id_granite_vision = "ibm/granite-vision-3-2-2b"
model_id_granite = "ibm/granite-3-2-8b-instruct"

# Global base parameters.
base_parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 10000,
    "min_new_tokens": 1,
    "temperature": 0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42  
}

def get_model_params(model_id):
    params = base_parameters.copy()
    if model_id == "ibm/granite-13b-instruct-v2":
        params["max_new_tokens"] = 8191
    return params

# Instantiate LLM objects with model-specific parameters.
llm_code = ChatWatsonx(
    model_id=model_id_code,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=get_model_params(model_id_code)
)

llm_mistral = ChatWatsonx(
    model_id=model_id_mistral,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=get_model_params(model_id_mistral)
)

llm_granite = ChatWatsonx(
    model_id=model_id_granite,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=get_model_params(model_id_granite)
)

llm_granite_vision = ChatWatsonx(
    model_id=model_id_granite_vision,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=get_model_params(model_id_granite_vision)
)

llm_llama = ChatWatsonx(
    model_id=model_id_llama,
    url=url,
    apikey=apikey,
    project_id=project_id,
    params=get_model_params(model_id_llama)
)

llm_chat_gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=openai_apikey
)

# Use MPS if available, otherwise use CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(2)

# Override print to capture output.
output_lines = []
_original_print = print
def custom_print(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)
    _original_print(message, **kwargs)
    output_lines.append(message)
print = custom_print

# Set up environment and disable tokenizer parallelism.
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

###############################################################################
# Global top_k retrieval variable.
###############################################################################
RETRIEVAL_TOP_K = 20
###############################################################################

# --- Helper Functions (for non-LLM operations) ---

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0.0

class FixedSizeSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    def split(self, text):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

def get_selected_model(llm):
    return getattr(llm, "model", None) or getattr(llm, "model_id", None)

# --- LLM Functions using llm.invoke(...).content ---

async def generate_summary(llm, text):
    prompt = f"Summarize the following text for global sensemaking:\n\n{text}\n\nSummary (return only the summary):"
    response = llm.invoke(prompt)
    return response.content.strip()

async def generate_partial_answer(llm, community_summary, query):
    prompt = f"Given the following community summary:\n\n{community_summary}\n\nAnswer the query: {query}\n\nPartial Answer (return only the answer):"
    response = llm.invoke(prompt)
    return response.content.strip()

async def generate_global_answer(llm, partial_answers, query):
    combined_text = "\n".join(partial_answers)
    prompt = f"Given the following partial answers:\n\n{combined_text}\n\nGenerate a final, coherent answer to the query: {query}\n\nFinal Answer (return only the final answer):"
    response = llm.invoke(prompt)
    return response.content.strip()

async def generate_answer(llm, context, question):
    prompt = f"Using the following context, answer the question succinctly.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response.content.strip()

async def evaluate_answer(llm, candidate_answer, correct_answer):
    prompt = f"""
Compare the given answer to the reference answer.
Return '1' if they are semantically similar, and '0' only if they are completely unrelated.

Reference Answer:
{correct_answer}

Candidate Answer:
{candidate_answer}

Judgment (only return '1' or '0'):
"""
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    match = re.search(r'\b([01])\b', response_text)
    if match:
        return int(match.group(1))
    else:
        return 0

async def load_document(file_path):
    print(f"📌 Loading document: {file_path}")
    file_extension = file_path.lower().split('.')[-1]
    if file_extension == "pdf":
        try:
            import pymupdf as fitz
            return await asyncio.to_thread(lambda: "\n".join(page.get_text("text") for page in fitz.open(file_path)))
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")
    else:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return (await f.read()).strip()
        except Exception as e:
            raise ValueError(f"Error reading text file: {e}")

def store_chunks_in_neo4j(driver, chunks, embeddings):
    with driver.session() as session:
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            session.run(
                "CREATE (c:Chunk {chunk_id: $id, text: $text, embedding: $embedding})",
                id=i, text=text, embedding=emb_list
            )
    print(f"✅ Stored {len(chunks)} chunks in Neo4j.")

# --- Microsoft Graph RAG Implementation ---

class MicrosoftGraphRAG:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = []
        self.graph = None
        self.community_summaries = []

    async def build_graph_index(self, document_text):
        print("📌 Building Graph RAG index using Microsoft’s approach...")
        splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=10)
        self.chunks = splitter.split(document_text)
        print(f"✅ Split document into {len(self.chunks)} chunks.")
        self.embeddings = []
        for chunk in self.chunks:
            emb = self.embedding_model.model.encode(chunk)
            self.embeddings.append(emb)
        expected_dim = self.embedding_model.model.get_sentence_embedding_dimension()
        print(f"✅ Generated embeddings (dimension: {expected_dim}).")
        num_nodes = len(self.chunks)
        self.graph = igraph.Graph()
        self.graph.add_vertices(num_nodes)
        threshold = 0.8
        edges = []
        weights = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = cosine_similarity(self.embeddings[i], self.embeddings[j])
                if sim > threshold:
                    edges.append((i, j))
                    weights.append(sim)
        if edges:
            self.graph.add_edges(edges)
            self.graph.es["weight"] = weights
            print(f"✅ Added {len(edges)} edges based on similarity threshold {threshold}.")
        else:
            print("⚠️ No edges added – consider lowering the similarity threshold.")
        partition = leidenalg.find_partition(self.graph, leidenalg.RBConfigurationVertexPartition)
        num_communities = len(partition)
        print(f"✅ Detected {num_communities} communities.")
        self.community_summaries = []
        for idx, community in enumerate(partition):
            community_text = "\n".join([self.chunks[i] for i in community])
            summary = await generate_summary(self.llm, community_text)
            self.community_summaries.append(summary)
            print(f"✅ Generated summary for community {idx+1}/{num_communities}.")

    async def search(self, query):
        print(f"📌 Generating partial answers for the query: {query}")
        partial_answers = []
        for idx, summary in enumerate(self.community_summaries):
            partial = await generate_partial_answer(self.llm, summary, query)
            partial_answers.append(partial)
            print(f" - Partial answer from community {idx+1}: {partial[:100]}...")
        global_answer = await generate_global_answer(self.llm, partial_answers, query)
        print("✅ Generated global answer.")
        return global_answer

# --- HYBRID RAG Implementation ---

class HybridRAG:
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = []
        self.graph = None
        self.community_summaries = []

    async def build_graph_index(self, retrieved_chunks):
        print("📌 Building Hybrid RAG index using top-k retrieved chunks...")
        self.chunks = retrieved_chunks
        print(f"✅ Received {len(self.chunks)} chunks from Vector DB for Hybrid RAG.")
        self.embeddings = []
        for chunk in self.chunks:
            emb = self.embedding_model.model.encode(chunk)
            self.embeddings.append(emb)
        expected_dim = self.embedding_model.model.get_sentence_embedding_dimension()
        print(f"✅ Generated embeddings for Hybrid RAG (dimension: {expected_dim}).")
        num_nodes = len(self.chunks)
        self.graph = igraph.Graph()
        self.graph.add_vertices(num_nodes)
        threshold = 0.8
        edges = []
        weights = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                sim = cosine_similarity(self.embeddings[i], self.embeddings[j])
                if sim > threshold:
                    edges.append((i, j))
                    weights.append(sim)
        if edges:
            self.graph.add_edges(edges)
            self.graph.es["weight"] = weights
            print(f"✅ [Hybrid RAG] Added {len(edges)} edges with similarity threshold {threshold}.")
        else:
            print("⚠️ [Hybrid RAG] No edges added – consider lowering the similarity threshold.")
        partition = leidenalg.find_partition(self.graph, leidenalg.RBConfigurationVertexPartition)
        num_communities = len(partition)
        print(f"✅ [Hybrid RAG] Detected {num_communities} communities.")
        self.community_summaries = []
        for idx, community in enumerate(partition):
            community_text = "\n".join([self.chunks[i] for i in community])
            summary = await generate_summary(self.llm, community_text)
            self.community_summaries.append(summary)
            print(f"✅ [Hybrid RAG] Generated summary for community {idx+1}/{num_communities}.")

    async def search(self, query):
        print(f"📌 [Hybrid RAG] Generating partial answers for the query: {query}")
        partial_answers = []
        for idx, summary in enumerate(self.community_summaries):
            partial = await generate_partial_answer(self.llm, summary, query)
            partial_answers.append(partial)
            print(f" - [Hybrid RAG] Partial answer from community {idx+1}: {partial[:100]}...")
        global_answer = await generate_global_answer(self.llm, partial_answers, query)
        print("✅ [Hybrid RAG] Generated global answer.")
        return global_answer

# --- Vector Database RAG Implementation ---
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.indexes import create_vector_index

# --- Main Function ---

async def main():
    document_path = "./name_of_pdf.pdf"
    csv_path = "./name_of_csv.csv"
    index_name = "graph_rag_index"

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    
    with driver.session() as session:
        session.run("MATCH (c:Chunk) DETACH DELETE c")
        session.run(f"DROP INDEX {index_name} IF EXISTS")

    llm_options = {
        "code": llm_code,
        "mistral": llm_mistral,
        "granite": llm_granite,
        "granite_vision": llm_granite_vision,
        "llama": llm_llama,
        "chat_gpt": llm_chat_gpt,
    }
    selected_model = os.getenv("SELECTED_MODEL", "granite")
    llm = llm_options.get(selected_model, llm_granite)
    
    embedding_model = SentenceTransformerEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model.model.to(device)
    
    microsoft_graphrag = MicrosoftGraphRAG(llm, embedding_model)
    document_text = await load_document(document_path)
    await microsoft_graphrag.build_graph_index(document_text)
    store_chunks_in_neo4j(driver, microsoft_graphrag.chunks, microsoft_graphrag.embeddings)
    dims = embedding_model.model.get_sentence_embedding_dimension()
    create_vector_index(
        driver,
        index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=dims,
        similarity_fn="cosine"
    )
    vector_rag = VectorRetriever(driver, index_name, embedding_model)

    graph_rag_score, vector_rag_score, hybrid_rag_score, total_questions = 0, 0, 0, 0

    async with aiofiles.open(csv_path, newline='', encoding='utf-8') as csvfile:
        content = await csvfile.read()
        lines = [line for line in content.splitlines() if line.strip()]
        rows = list(csv.DictReader(lines))
        total_questions = len(rows)
        counter = 0
        for row in rows:
            counter += 1
            question, correct_answer = row["question"], row["reference"]

            graph_rag_answer = await microsoft_graphrag.search(question)

            vr_results = vector_rag.search(query_text=question, top_k=RETRIEVAL_TOP_K)
            vr_raw_context = ""
            for chunk in vr_results:
                if isinstance(chunk, dict) and 'text' in chunk and chunk['text'].strip():
                    vr_raw_context += chunk['text'].strip() + "\n"
            if not vr_raw_context.strip():
                vr_raw_context = "[No answer returned]"
            vector_rag_answer = await generate_answer(llm, vr_raw_context, question)

            hr_results = vector_rag.search(query_text=question, top_k=RETRIEVAL_TOP_K)
            hr_chunks = []
            for chunk in hr_results:
                if isinstance(chunk, dict) and 'text' in chunk and chunk['text'].strip():
                    hr_chunks.append(chunk['text'].strip())
            hybrid_rag = HybridRAG(llm, embedding_model)
            await hybrid_rag.build_graph_index(hr_chunks)
            hybrid_rag_answer = await hybrid_rag.search(question)

            print("\n==========================")
            print(f"Question: {counter}/{total_questions} - {question}")
            print("Microsoft Graph RAG Answer:")
            print(graph_rag_answer)
            print()
            print("Vector Database RAG Generated Answer:")
            print(vector_rag_answer)
            print()
            print("[HYBRID RAG] Final Answer (GraphRAG over top-k retrieved chunks):")
            print(hybrid_rag_answer)
            print(f"Current Scores -> Microsoft Graph RAG: {graph_rag_score} | Vector DB RAG: {vector_rag_score} | Hybrid RAG: {hybrid_rag_score}")
            print("==========================\n")

            graph_rag_score += await evaluate_answer(llm, graph_rag_answer, correct_answer)
            vector_rag_score += await evaluate_answer(llm, vector_rag_answer, correct_answer)
            hybrid_rag_score += await evaluate_answer(llm, hybrid_rag_answer, correct_answer)

            print(f"Updated Scores -> Microsoft Graph RAG: {graph_rag_score} | Vector DB RAG: {vector_rag_score} | Hybrid RAG: {hybrid_rag_score}\n")

    print(f"\n✅ Microsoft Graph RAG Final Score: {graph_rag_score}/{total_questions}")
    print(f"🔵 Vector Database RAG Final Score: {vector_rag_score}/{total_questions}")
    print(f"🟣 Hybrid RAG Final Score: {hybrid_rag_score}/{total_questions}")

    final_output = "\n".join(output_lines)
    with open("GRAPHRAGvsRAGvsHYBRID.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    print(final_output)
    driver.close()

if __name__ == "__main__":
    asyncio.run(main())