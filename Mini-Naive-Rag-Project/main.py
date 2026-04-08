import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from  langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv(dotenv_path="D:\GenAI_Langchain\Agentic_AI\RAG_implementation\.env")

print("initializing components....")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5-mini", api_key=os.getenv("OPENAI_API_KEY"))

vectostore = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"), embedding= embeddings, pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

retreiver = vectostore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the questions based on the following content:
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Foramt Retreived documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

def retreival_chain_without_LCEL(Query:str):
    """Simple Retreival chain"""
    docs = retreiver.invoke(Query)
    context = format_docs(docs)

    message = prompt_template.format_messages(
        context=context,question=Query
    )

    response = llm.invoke(message)

    return response.content


def create_retreival_chain_with_lcel():
    retreival_chain = (RunnablePassthrough.assign(
        context=itemgetter("question") | retreiver | format_docs 
        )
    | prompt_template 
    | llm 
    | StrOutputParser() 
    )
    return retreival_chain


if __name__ == "__main__":
    query = "What is pinecone in machine Learning ?"
    # print("retreiving.....")
    # print("\n\n"+ "=" * 70)
    # result_without_lcel = retreival_chain_without_LCEL(query)
    # print("\nANswer")
    # print(result_without_lcel)

    print("Retreiving with lcel.....")
    print("\n\n"+ "=" * 70)
    result_with_lcel = create_retreival_chain_with_lcel()
    response = result_with_lcel.invoke({"question":query})
    print("\nAnswer")
    print(response)
    

    

