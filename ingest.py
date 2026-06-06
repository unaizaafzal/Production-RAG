from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_chunk_docs(data_dir: str, chunk_size: int = 1200, chunk_overlap: int = 300):
    """
    Loads all PDFs in a directory using PyPDFium2 to accurately extract 
    text from multi-column academic and technical paper structures.
    """
    print(f"--> Loading PDFs from: {data_dir}")
    all_docs = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory '{data_dir}' does not exist.")
        return []
        
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_dir, file)
            print(f"--> Parsing: {file}")
            loader = PyPDFium2Loader(file_path)
            all_docs.extend(loader.load())
            
    print(f"--> Successfully extracted {len(all_docs)} pages.")
    
    # Advanced structural recursive splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_docs)
    print(f"--> Created {len(chunks)} structural context chunks.")
    return chunks

def index_to_pinecone_native(chunks, index_name: str):
    """
    Directly loops and upserts dense text vector embeddings into a clean Pinecone index.
    """
    print("--> Initializing Hugging Face Embeddings (all-MiniLM-L6-v2)...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Establishes connection using environment variables
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    
    print(f"--> Generating vectors and upserting {len(chunks)} chunks to Pinecone...")
    
    upsert_data = []
    for idx, chunk in enumerate(chunks):
        # Calculate local raw vectors
        vector = embeddings_model.embed_query(chunk.page_content)
        
        # Isolate clean source references
        source_path = chunk.metadata.get("source", "Unknown")
        page_num = chunk.metadata.get("page", 0)
        
        # Build vector records payload matching native formatting
        upsert_data.append({
            "id": f"chunk_{idx}",
            "values": vector,
            "metadata": {
                "text": chunk.page_content,
                "source": source_path,
                "page": page_num
            }
        })
        
        # Batch upload to Pinecone in groups of 25 for absolute safety against size caps
        if len(upsert_data) == 25 or idx == len(chunks) - 1:
            index.upsert(vectors=upsert_data)
            upsert_data = []
            
    print("--> Success! Vectors have been natively indexed in Pinecone.")

if __name__ == "__main__":
    # The key is now securely fetched from memory automatically!
    INDEX_NAME = "production-rag" 
    DATA_DIRECTORY = "./data"

    chunks = load_and_chunk_docs(DATA_DIRECTORY, chunk_size=1200, chunk_overlap=300)

    if chunks:
        index_to_pinecone_native(chunks, INDEX_NAME)