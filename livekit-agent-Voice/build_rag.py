import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_knowledge_base():
    print("Reading knowledge.txt...")
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print("Error: knowledge.txt not found. Cannot build RAG.")
        return

    print("Chunking document...")
    # Break text into 300 character chunks with 50 chars of overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])
    
    print(f"Generated {len(chunks)} chunks.")

    print("Initializing local Vector Store (ChromaDB)...")
    # This will create a `.chroma_db` folder locally
    client = chromadb.PersistentClient(path="./.chroma_db")
    
    # We delete the old collection if it exists to refresh the embeddings
    try:
        client.delete_collection(name="voice_agent_kb")
    except Exception:
        pass
        
    collection = client.create_collection(name="voice_agent_kb")
    
    # Prepare ids and documents
    ids = [str(i) for i in range(len(chunks))]
    documents = [chunk.page_content for chunk in chunks]

    print("Generating embeddings and writing to DB...")
    # Chroma uses SentenceTransformers automatically unless you provide an embedding_function.
    collection.add(
        documents=documents,
        ids=ids
    )
    
    print("RAG Knowledge Base successfully built!")

if __name__ == "__main__":
    build_knowledge_base()
