import os
import time
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import numpy as np
import cohere
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse import get_client

load_dotenv()

# Initialize core clients
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc_index = pc.Index("production-rag")
co_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
langfuse = get_client()

# --- BENCHMARK GOLDEN EVALUATION DATASET ---
GOLDEN_DATASET = [
    {
        "question": "What is a Grand Slam Offer?",
        "expected_ground_truth": "An offer you make people so good they would feel stupid saying no.",
        "namespace": "ns_2b85a157" # Uses your active indexed namespace
    }
]

def run_eval_retrieval(query: str, namespace: str, top_n: int = 4):
    """Bypasses Streamlit state to pull contexts strictly from a specified Pinecone namespace."""
    query_vector = embeddings_model.embed_query(query)
    dense_response = pc_index.query(vector=query_vector, top_k=15, include_metadata=True, namespace=namespace)
    
    documents = []
    for match in dense_response.get("matches", []):
        if "text" in match["metadata"]:
            documents.append(match["metadata"]["text"])
            
    if not documents:
        return "No context found."
        
    # Cloud Rerank
    response = co_client.rerank(model="rerank-v3.5", query=query, documents=documents, top_n=top_n)
    context_chunks = [documents[res.index] for res in response.results]
    return "\n\n---\n\n".join(context_chunks)

def judge_faithfulness(context: str, answer: str) -> float:
    """LLM-as-a-judge score evaluating if the answer is fully grounded in the retrieved text (0.0 to 1.0)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an unbiased AI quality auditor. Rate the FAITHFULNESS of the provided answer using ONLY the context given. If the answer contains information not explicitly supported by the context, it is a hallucination. Respond with a single raw JSON object containing exactly two keys: 'reasoning' (string) and 'score' (float between 0.0 and 1.0). Do not include markdown codeblocks."),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        result = json.loads(chain.invoke({"context": context, "answer": answer}).strip().replace("```json", "").replace("```", ""))
        return float(result.get("score", 0.0))
    except Exception:
        return 0.5

def run_automated_evaluation_suite():
    print(" Initializing Automated AI Quality Auditing Framework...")
    all_passed = True
    
    for case in GOLDEN_DATASET:
        print(f"\nEvaluating Question: '{case['question']}'")
        
        # 1. Fetch Context
        context = run_eval_retrieval(case["question"], case["namespace"])
        
        # 2. Generate System Answer
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the provided context:\n\n{context}"),
            ("human", "{question}")
        ])
        chain = qa_prompt | llm | StrOutputParser()
        generated_answer = chain.invoke({"context": context, "question": case["question"]})
        
        # 3. Grade using LLM-as-a-judge
        # 3. Grade using LLM-as-a-judge
        faithfulness_score = judge_faithfulness(context, generated_answer)
        print(f"✨ LLM-as-a-Judge Faithfulness Score: {faithfulness_score * 100}%")
        
        # Log evaluation metrics directly into Langfuse using the modern Python SDK method
        eval_trace_id = f"eval_{int(time.time())}"
        langfuse.create_score(
            trace_id=eval_trace_id,
            name="automated-faithfulness",
            value=faithfulness_score,
            comment=f"Automated verification for question: {case['question']}"
        )
        
        # Performance Threshold Gate (e.g., must be higher than 80% grounded)
        if faithfulness_score < 0.8:
            print(" REGRESSION DETECTED: Faithfulness score dropped below target enterprise threshold!")
            all_passed = False
        
        # Log evaluation metrics directly into Langfuse for persistent engineering visibility
        # Log evaluation metrics directly into Langfuse using the modern Python SDK method
        # We pass a generated or associated trace ID to attach the metric cleanly
        eval_trace_id = f"eval_{int(time.time())}"
        langfuse.create_score(
            trace_id=eval_trace_id,
            name="automated-faithfulness",
            value=faithfulness_score,
            comment=f"Automated verification for question: {case['question']}"
        )
        
        # Performance Threshold Gate (e.g., must be higher than 80% grounded)
        if faithfulness_score < 0.8:
            print(" REGRESSION DETECTED: Faithfulness score dropped below target enterprise threshold!")
            all_passed = False
            
    if all_passed:
        print("\n All quality gates passed perfectly. System verified for release.")
        exit(0)
    else:
        print("\n Deployment blocked due to quality regressions.")
        exit(1)

if __name__ == "__main__":
    run_automated_evaluation_suite()