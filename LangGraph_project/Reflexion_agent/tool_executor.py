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

load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from schema import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)

def run_queries(search_queries: list[str], **kwargs):
    """Run The generated queries"""

    return tavily_tool.batch([{"query": query} for query in search_queries])

exec_tools =ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__)
    ]
)

