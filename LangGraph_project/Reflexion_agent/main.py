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


from typing import Literal

from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import AIMessage, ToolMessage
from dotenv import load_dotenv

from chains import first_responder, reviser
from tool_executor import exec_tools

MAX_ITER = 2

load_dotenv()

def draft_node(state:MessagesState):
    """Draft the initial response."""

    res = first_responder.invoke({"messages":state["messages"]})
    return {"messages": [res]}

def revise_node(state:MessagesState):
    """Revise the answer based on the search tool results"""
    res = reviser.invoke({"messages": state["messages"]})
    return {"messages":[res]}

def event_loop(state:MessagesState) -> Literal["execute_tools", END]:
    """Determine Whether to continue or end based on iteration count."""

    count_tool_calls = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )

    num_iter = count_tool_calls
    if num_iter > MAX_ITER:
        return END
    return "execute_tools"

reflexion_graph = StateGraph(MessagesState)
reflexion_graph.add_node("draft", draft_node)
reflexion_graph.add_node("execute_tools", exec_tools)
reflexion_graph.add_node("revise",revise_node)
reflexion_graph.add_edge(START, "draft")
reflexion_graph.add_edge("draft", "execute_tools")
reflexion_graph.add_edge("execute_tools", "revise")
reflexion_graph.add_conditional_edges("revise", event_loop, {"execute_tools", END})

graph = reflexion_graph.compile()


res = graph.invoke(
    {
    "messages": [
        {
        "role": "user",
        "content": "Write about Ai-powered SOC / autonomous SOC problem domain, List Startups that do that and raised capital."
        }
        
        ]
  }
)

last_messge = res["messages"][-1]
if isinstance(last_messge, AIMessage) and last_messge.tool_calls:
    print(last_messge.tool_calls[0]["args"]["answer"])



