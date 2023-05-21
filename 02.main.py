# Using LlamaIndex + OpenAI
import os
from dotenv import load_dotenv
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ResponseSynthesizer
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor

load_dotenv()
documents = SimpleDirectoryReader('Docs').load_data() #load coduments
index = GPTVectorStoreIndex.from_documents(documents) #index documents
index.storage_context.persist(persist_dir='indexes') # store indexes
storage_context = StorageContext.from_defaults(persist_dir="indexes") #set storage context
index = load_index_from_storage(storage_context) #used stored index

retriever = VectorIndexRetriever(
    index=index, 
    similarity_top_k=2, #top 2 results
)

response_synthesizer = ResponseSynthesizer.from_args(
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.75) #75% match
    ]
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

query_engine = index.as_query_engine()

def query(q):
    print("Q: ", q)
    a = query_engine.query(q)
    print(a)

query("What is Langchain?")
query("What is llamaindex?")
query("What is the difference between Langchain and llamaindex?")
query("What is Policy Driven Governance?")
#query("What are Azure Landing Zones?, Is it Scalable ?")
#query("What is Azure Platform Landing Zone?")
#query("What are Azure Functions used for?")