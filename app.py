import os
import tempfile
import uuid
import time
import numpy as np
import cohere
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
# Modern clean Langfuse import paths 
# Modern clean Langfuse SDK imports (Wipes out the ModuleNotFoundError)
from langfuse import observe, propagate_attributes, get_client
from langfuse.langchain import CallbackHandler

# Inject keys from hidden secure .env file
load_dotenv()

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_title="Production RAG Engine", page_icon="🟥", layout="wide")

# --- INITIALIZE SHARED CACHED SERVICES ---
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

embeddings_model = load_embedding_model()
llm = get_llm()

# Initialize API clients
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "production-rag"
pc_index = pc.Index(index_name)
co_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

# --- SESSION STATE INITIALIZATION ---
if "session_id" not in st.session_state:
    
    st.session_state.session_id = f"ns_{uuid.uuid4().hex[:8]}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False

# Operational Latency Trackers for UI
if "p50_latency" not in st.session_state:
    st.session_state.p50_latency = 0.0
if "p95_latency" not in st.session_state:
    st.session_state.p95_latency = 0.0
if "latency_history" not in st.session_state:
    st.session_state.latency_history = []

NAMESPACE = st.session_state.session_id

# --- CORE RAG PROCESSING LOGIC ---
def process_and_index_pdf(uploaded_file):
    """Saves the uploaded file stream to a temporary file, chunks it, and indexes to Pinecone."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFium2Loader(tmp_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])
        chunks = text_splitter.split_documents(docs)
        
        upsert_data = []
        for idx, chunk in enumerate(chunks):
            vector = embeddings_model.embed_query(chunk.page_content)
            upsert_data.append({
                "id": f"chunk_{idx}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "Uploaded File"),
                    "page": chunk.metadata.get("page", 0)
                }
            })
            
            if len(upsert_data) == 25 or idx == len(chunks) - 1:
                pc_index.upsert(vectors=upsert_data, namespace=NAMESPACE)
                upsert_data = []
                
        return chunks
    finally:
        os.remove(tmp_path)

def contextualize_query(user_query: str, chat_history: list, langfuse_handler: CallbackHandler) -> str:
    """Reformulates latest user text into a standalone search term using conversation memory."""
    if not chat_history:
        return user_query
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question which can be searched in a vector database. Do NOT answer it, just rewrite it if needed, otherwise return as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    chain = contextualize_prompt | llm | StrOutputParser()
    # Route intermediate generation callbacks to Langfuse handler
    return chain.invoke({"chat_history": chat_history, "question": user_query}, config={"callbacks": [langfuse_handler]})

def execute_hybrid_retrieval(query: str, chunks: list, top_n_to_llm: int = 4):
    """Executes fused keyword/dense lookup strictly bounded inside the session namespace with modern observations."""
    chunk_texts = [chunk.page_content for chunk in chunks]
    langfuse = get_client()
    
    # 1. Sparse BM25 Execution Span
    with langfuse.start_as_current_observation(as_type="span", name="bm25-sparse-retrieval", input={"query": query}) as span:
        tokenized_corpus = [doc.lower().split(" ") for doc in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranked_ids = [f"chunk_{idx}" for idx in np.argsort(bm25_scores)[::-1]]
        span.update(output={"top_bm25_id": bm25_ranked_ids[0] if bm25_ranked_ids else None})
    
    # 2. Pinecone Vector Execution Span
    with langfuse.start_as_current_observation(as_type="span", name="pinecone-dense-retrieval", input={"query": query, "namespace": NAMESPACE}) as span:
        query_vector = embeddings_model.embed_query(query)
        dense_response = pc_index.query(vector=query_vector, top_k=len(chunks), include_metadata=False, namespace=NAMESPACE)
        vector_ranked_ids = [match["id"] for match in dense_response.get("matches", [])]
        span.update(output={"match_count": len(vector_ranked_ids)})
    
    # Reciprocal Rank Fusion
    rrf_scores = {}
    for rank, match_id in enumerate(vector_ranked_ids): rrf_scores[match_id] = rrf_scores.get(match_id, 0.0) + 1.0 / (rank + 60)
    for rank, match_id in enumerate(bm25_ranked_ids): rrf_scores[match_id] = rrf_scores.get(match_id, 0.0) + 1.0 / (rank + 60)
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    documents_to_rerank = []
    chunk_mapping = []
    for chunk_id, _ in fused_results[:15]:
        idx = int(chunk_id.split("_")[1])
        documents_to_rerank.append(chunks[idx].page_content)
        chunk_mapping.append(chunks[idx])
        
    # 3. Cohere Cloud Reranking Generation Span
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="cohere-cloud-rerank", 
        model="rerank-v3.5",
        input={"query": query, "documents_count": len(documents_to_rerank)}
    ) as gen_span:
        response = co_client.rerank(model="rerank-v3.5", query=query, documents=documents_to_rerank, top_n=top_n_to_llm)
        gen_span.update(output={"reranked_results": [{"index": r.index, "score": r.relevance_score} for r in response.results]})
    
    context_chunks = []
    for result in response.results:
        selected_chunk = chunk_mapping[result.index]
        context_chunks.append(f"[Source: {selected_chunk.metadata.get('source')}, Page: {selected_chunk.metadata.get('page', 0)}]\n{selected_chunk.page_content}")
        
    return "\n\n---\n\n".join(context_chunks)

# --- WRAP THE PRIMARY TRANSACTION WITH @OBSERVE ---
@observe(name="rag-chat-transaction")
def run_monitored_rag_cycle(user_input, chunks):
    """Executes the contextualization, search, and answer generation loops as a single trace."""
    # Retrieve the active root client and register the LangChain handler
    langfuse = get_client()
    langfuse_handler = CallbackHandler()
    
    # Securely propagate session details and custom metadata across the trace execution scope
    with propagate_attributes(
        session_id=st.session_state.session_id,
        user_id="unaiza_afzal",
        tags=["production", "streamlit-frontend"]
    ):
        # Step A: Contextualize
        standalone_search = contextualize_query(user_input, st.session_state.chat_history, langfuse_handler)
        
        # Step B: Retrieval
        context_string = execute_hybrid_retrieval(standalone_search, chunks)
        
        # Step C: Answer Generation
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical assistant analyzing specialized documentation. Answer the question using ONLY the provided context below. If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n\nContext:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = qa_prompt | llm | StrOutputParser()
        ai_response = chain.invoke({
            "context": context_string,
            "chat_history": st.session_state.chat_history,
            "question": user_input
        }, config={"callbacks": [langfuse_handler]})
        
        return ai_response
    
    # Step A: Contextualize
    standalone_search = contextualize_query(user_input, st.session_state.chat_history, langfuse_handler)
    
    # Step B: Retrieval
    context_string = execute_hybrid_retrieval(standalone_search, chunks)
    
    # Step C: Answer Generation
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert technical assistant analyzing specialized documentation. Answer the question using ONLY the provided context below. If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = qa_prompt | llm | StrOutputParser()
    ai_response = chain.invoke({
        "context": context_string,
        "chat_history": st.session_state.chat_history,
        "question": user_input
    }, config={"callbacks": [langfuse_handler]})
    
    return ai_response

# --- UI VISUAL LAYOUT ---
st.title("Enterprise Multi-Tenant RAG Platform")
st.markdown("An advanced layout-aware search pipeline featuring hybrid RRF retrieval, cloud reranking, and secure session namespace isolation.")

# Sidebar Controls for document updates
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload a technical document (PDF)", type=["pdf"])
    
    if uploaded_file:
        if st.button("Ingest & Index Document", use_container_width=True):
            with st.spinner("Executing structural chunking and vector indexing..."):
                st.session_state.chunks = process_and_index_pdf(uploaded_file)
                st.session_state.ingested = True
                st.session_state.chat_history = []  
                st.success(f"Successfully indexed {len(st.session_state.chunks)} chunks into Namespace: {NAMESPACE}")
                
    st.write("---")
    st.header("Real-Time Telemetry")
    st.metric("p50 (Median Latency)", f"{st.session_state.p50_latency:.2f}s")
    st.metric("p95 (Tail Latency)", f"{st.session_state.p95_latency:.2f}s")
    st.caption(f"**Active Namespace ID:** `{NAMESPACE}`")

# Main Interface Screen Routing
if not st.session_state.ingested:
    st.info("Welcome! Please upload and ingest a PDF document using the sidebar panel to unlock the conversational AI engine.")
else:
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

    if user_input := st.chat_input("Ask a specialized question about your document..."):
        st.chat_message("user").write(user_input)
        
        with st.spinner("Analyzing context windows and cloud ranking..."):
            start_time = time.time()
            
            # Fire the completely observed transaction cycle
            ai_response = run_monitored_rag_cycle(user_input, st.session_state.chunks)
            
            duration = time.time() - start_time
            
            # Dynamically calculate p50 and p95 distributions in-memory
            st.session_state.latency_history.append(duration)
            latencies = np.array(st.session_state.latency_history)
            st.session_state.p50_latency = np.percentile(latencies, 50)
            st.session_state.p95_latency = np.percentile(latencies, 95)
            
            st.chat_message("assistant").write(ai_response)
            # Store updates back into tracking records
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=ai_response))
            
            # Force a re-render to instantly update sidebar metrics
            st.rerun()