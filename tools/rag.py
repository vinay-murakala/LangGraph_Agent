import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "resources/AI_Agents.pdf"
QDRANT_PATH = "./qdrant_data"
COLLECTION = "ai_agents"
EMBEDDING_MODEL = "models/text-embedding-004"

def get_retriever(client):
    '''Create a retriever based on input doc and embedding model defined'''

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL,google_api_key=os.environ["GOOGLE_API_KEY"])


    # Fetch the existing Qdrant store for given collection if it already exists in database
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION in collection_names:
        print(f"Loaded existing Qdrant collection: {COLLECTION}")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=embeddings,
        )
        return vector_store.as_retriever(search_kwargs={"k": 3})
        
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )
    
    # Created Vector store with give document and generated splits
    vector_store.add_documents(documents=splits)
    
    # Retrieve top k(3) samples
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":

    client = QdrantClient(path=QDRANT_PATH)

    try:
        retriever = get_retriever(client)
        print("Test retrieval query: 'Summary'")
        results = retriever.invoke("Give me a summary")
        print(f"Found {len(results)} relevant chunks.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Explicitly close the client so __del__ doesn't run at shutdown
        client.close()