import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv


load_dotenv()

if __name__ == '__main__':
    print("Ingesting Text File....")
    loader = TextLoader("D:\GenAI_Langchain\Agentic_AI\RAG_implementation\mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    print("Ingesting into vector store")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Finished Ingesting...")
    








