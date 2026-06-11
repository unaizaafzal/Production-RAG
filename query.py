from dotenv import load_dotenv
load_dotenv()
import os
import numpy as np
import cohere
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Import document loader from your working ingest.py
from ingest import load_and_chunk_docs

def reciprocal_rank_fusion(vector_results, bm25_results, c=60):
    """Blends dense and sparse rankings."""
    rrf_scores = {}
    for rank, match_id in enumerate(vector_results):
        rrf_scores[match_id] = rrf_scores.get(match_id, 0.0) + 1.0 / (rank + c)
    for rank, match_id in enumerate(bm25_results):
        rrf_scores[match_id] = rrf_scores.get(match_id, 0.0) + 1.0 / (rank + c)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def contextualize_query(user_query: str, chat_history: list, llm: ChatGroq) -> str:
    """
    If there is chat history, reformulates the latest user query into a 
    standalone question that can be searched in the vector database.
    """
    if not chat_history:
        return user_query
        
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = contextualize_prompt | llm | StrOutputParser()
    standalone_query = chain.invoke({"chat_history": chat_history, "question": user_query})
    print(f"Contextualized Search Query: '{standalone_query}'")
    return standalone_query

def retrieve_hybrid_and_cohere_rerank(query: str, index_name: str, chunks: list, pc_index, embeddings_model, co_client, top_n_to_llm: int = 4):
    """Executes the complete retrieval pipeline using an already loaded corpus."""
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # 1. Local BM25 Keyword Search
    tokenized_corpus = [doc.lower().split(" ") for doc in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked_ids = [f"chunk_{idx}" for idx in np.argsort(bm25_scores)[::-1]]
    
    # 2. Pinecone Dense Vector Search
    query_vector = embeddings_model.embed_query(query)
    dense_response = pc_index.query(vector=query_vector, top_k=len(chunks), include_metadata=False)
    vector_ranked_ids = [match["id"] for match in dense_response.get("matches", [])]
    
    # 3. Merge results using RRF (Pull top 15 chunks)
    fused_results = reciprocal_rank_fusion(vector_ranked_ids, bm25_ranked_ids, c=60)
    wide_pool_chunks = fused_results[:15] 
    
    # 4. Extract text strings and keep references for mapping
    documents_to_rerank = []
    chunk_mapping = []
    for chunk_id, _ in wide_pool_chunks:
        idx = int(chunk_id.split("_")[1])
        documents_to_rerank.append(chunks[idx].page_content)
        chunk_mapping.append(chunks[idx])
        
    # 5. Cloud Reranking via Cohere API
    response = co_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents_to_rerank,
        top_n=top_n_to_llm
    )
    
    # 6. Compile Final Context Window
    context_chunks = []
    for result in response.results:
        selected_chunk = chunk_mapping[result.index]
        source = selected_chunk.metadata.get("source", "Unknown")
        page = selected_chunk.metadata.get("page", 0)
        context_chunks.append(f"[Source: {source}, Page: {page}]\n{selected_chunk.page_content}")
        
    return "\n\n---\n\n".join(context_chunks)

def start_chat_session():
    INDEX_NAME = "production-rag"
    DATA_DIRECTORY = "./data"
    
    # Initialize Core Clients Once for the Session
    print("--> Initializing Shared Services & Loading Document Corpus...")
    chunks = load_and_chunk_docs(DATA_DIRECTORY)
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pc_index = pc.Index(INDEX_NAME)
    co_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
    
    # This array stores our chat messages during runtime
    chat_history = []
    
    print("\n RAG Chat Session Active! Type 'exit' or 'quit' to end.")
    print("==========================================================")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Ending session. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        try:
            # Step A: Contextualize the query based on past history
            search_query = contextualize_query(user_input, chat_history, llm)
            
            # Step B: Fetch optimized context using the standalone query
            context_str = retrieve_hybrid_and_cohere_rerank(
                search_query, INDEX_NAME, chunks, pc_index, embeddings_model, co_client, top_n_to_llm=4
            )
            
            # Step C: Generate Answer keeping history intact
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an expert technical assistant analyzing highly specialized documentation.\n"
                    "Answer the user's question using ONLY the provided context below. "
                    "You MUST cite your sources. After each claim, reference the source like this: [Source: filename, Page: X]. "
                    "If the context does not contain the answer, say 'I cannot find the answer in the provided documents.'\n\n"
                    "Context:\n{context}"
                )),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            chain = qa_prompt | llm | StrOutputParser()
            response = chain.invoke({
                "context": context_str,
                "chat_history": chat_history,
                "question": user_input
            })
            
            print(f"\nAI: {response}")
            
            # Step D: Append to session history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))
            
            # Optional: Limit memory size to last 6 messages to keep processing lightning fast
            if len(chat_history) > 6:
                chat_history = chat_history[-6:]
                
        except Exception as e:
            print(f" An error occurred: {e}")

if __name__ == "__main__":
   
    start_chat_session()