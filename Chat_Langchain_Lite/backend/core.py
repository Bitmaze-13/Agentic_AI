import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv(dotenv_path="D:\GenAI_Langchain\Agentic_AI\RAG_Project_Documentation_Helper\.env")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorStore = PineconeVectorStore(
    index_name="documentation-helper", embedding=embeddings
)

model = init_chat_model("gpt-5-mini", model_provider="openai")

@tool(response_format="content_and_artifact")
def retreive_context(query:str):
    """Retreive context relevant to the documentation to help answer user question/queries """
    retreive_docs = vectorStore.as_retriever().invoke(query,k=5)

    #Serialized document for the model
    serialized = "\n\n".join(
        (f"Source:{doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}")
        for doc in retreive_docs
    )

    return serialized, retreive_docs

def run_llm(query:str) -> Dict[str, Any]:
    """
    Run RAG Pipeline to answer user query

    Args:
        query: user's Question
    
        Returns:
            Dictionary containing
                -answer: the generated answer.
                - context: List of retreived documents.
    """
    System_prompt = (
        "You are helpful ai assistant that answer questions about Langchain Documentation."
        "You have access to a tool that retreives relevant documentation"
        "Use the tool to find the relevant information before answering questions"
        "Always cite the sources you use in your answers."
        "if you cannot find the answer in the retreived documentation , say so." 
    )

    agent = create_agent(model, tools=[retreive_context], system_prompt=System_prompt)

    messages = [{"role":"user","content": query}]

    response = agent.invoke({"messages":messages})

    answer = response["messages"][-1].content

    content_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                content_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": content_docs
    }

if __name__ == "__main__":
    result = run_llm(query="what are deep agents")
    print(result)
