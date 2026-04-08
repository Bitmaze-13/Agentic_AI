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

from dotenv import load_dotenv
from  langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state:MessagesState) -> bool:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT



flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)


flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT,
} )

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello React Langgraph with Function Calling")
    res = app.invoke({"messages": HumanMessage(content="What is the weather in Tokyo right now? List it then triple it.")})
    print(res["messages"][LAST].content)