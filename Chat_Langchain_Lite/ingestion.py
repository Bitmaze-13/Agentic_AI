import asyncio
import os
import ssl
from typing import Any, Dict, List
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

import certifi
from tavily import TavilyClient
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_max_seconds=10
)

vectorStore = PineconeVectorStore(
    index_name="documentation-helper", embedding= embeddings
)

# tavily_extract = TavilyExtract()
# tavily_map = TavilyMap(max_depth= 5, max_breadth=20, max_pages=1000)
# tavily_crawl = TavilyCrawl()
tavily_client = TavilyClient() 

async def index_documents_async(documents:List[Document], batch_size: int=50):
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"VectorStore Indexing: Preparing to add {len(documents)} documents to vector store. ",
        Colors.DARKCYAN,
    )
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each."
    )

    async def add_batch(batch: List[Document], batch_num:int):
        try:
            await vectorStore.aadd_documents(batch)
            log_success(
                f"VectoreStore Indexing: Successfully added batch {batch_num}/{len(batches)} (len)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add Batch {batch_num} - {e}")
            return False

    
    # tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    # results = await asyncio.gather(*tasks, return_exceptions=True)

    # successful = sum(1 for result in results if result is True)
    successful = 0
    for i , batch in enumerate(batches):
        result = await add_batch(batch, i + 1)
        if result:
            successful += 1
        
        await asyncio.sleep(0.5)

    if successful == len(batches):
        log_success(
            f"VectoreStore indexing: All Batches processed successfully {successful}/{len(batches)}"
        )
    else:
        log_warning(
            f"VectoreStore Indexing: Processed {successful}/{len(batches)} batches successfully."
        )


async def main():
    """Main async Function to orchestrate the entire process."""

    log_header("DOCUMENTATION INGESTION PIPELINE.")

    log_info("TavilyCrawl: Starting to crawl documentation from https://python.langchain.com/",
             Colors.PURPLE,
            )
    
    res = tavily_client.crawl(
    url = "https://python.langchain.com/",
    extract_depth= "advanced",
    max_depth=5
    )

    # In the LangChain tool version, 'res' is often the list itself 
    # rather than a dictionary containing 'results'
    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result["url"]}) for result in res.get("results",[]) if result.get('raw_content') is not None ]

    print(f"Successfully crawled {len(all_docs)} URLs")

    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} URLS from documentation site"
    )

    log_header("DOCUMENTATION CHUNKING PHASE.")
    log_info(f"Text Splitter : Processing {len(all_docs)} document with 4000 chunk size and 200 overlap.",
             Colors.YELLOW,
            )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap= 200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    await index_documents_async(splitted_docs, batch_size=500)




if __name__ == "__main__":
    asyncio.run(main())