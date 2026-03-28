import os
import ssl
import requests
from urllib3.exceptions import InsecureRequestWarning

# 1. Suppress the annoying 'InsecureRequest' warnings in your console
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# 2. Monkey-patch the 'requests' library to globally disable verification
# This forces Tavily and LangChain to ignore SSL errors
_old_request = requests.Session.request
def new_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _old_request(self, method, url, **kwargs)
requests.Session.request = new_request

# 3. Standard Python SSL bypass (as a backup)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ""
os.environ['PYTHONHTTPSVERIFY'] = "0"
from typing import List

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()



tavily = TavilyClient()
@tool
def search(query:str) -> str:
    """
    Tool That searches internet.
    Args:
        query: user Query to search.
    Returns:
        the search result 
    """

    print(f"Searching for {query}")
    return tavily.search(query=query)


class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="the url of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with the answer and sources"""

    answer:str = Field(description="The agent's answer ")
    sources:List[Source] = Field(default_factory = list, description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-5-nano")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from react-agent!")
    result = agent.invoke({"messages":[HumanMessage(content="how is the weather in india Pune ? and use the tools available to you.")]})
    print(result["structured_response"])

if __name__ == "__main__":
    main()
