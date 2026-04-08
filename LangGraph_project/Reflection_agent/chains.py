from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="D:\GenAI_Langchain\Agentic_AI\LangGraph_project\.env")

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral infuencer grading tweet. Generate critique and recommendation for the user's tweet"
            "Always provide detailed recommendations, including requests for length , virality, style, etc. "

        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a twitter techie influencer assistant tasked with writing excellent twitter posts"
            "Generate the best twitter post for user's request."
            "If the User provide critique, respond with the revised version of the previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

generation_chain = generation_prompt | llm

reflection_chain = reflection_prompt | llm


