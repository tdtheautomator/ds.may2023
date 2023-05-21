# Using Langchain + OpenAI + FAISS
import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import OpenAI

load_dotenv()

embeddings = OpenAIEmbeddings()
loader = DirectoryLoader('Docs',glob="**/*.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def query(q):
    print("Q: ", q)
    a = qa.run(q)
    print("A: ", a)

query("What is Langchain?")
query("What is llamaindex?")
query("What is the difference between Langchain and llamaindex?")
query("What is Policy Driven Governance?")
#query("What are Azure Landing Zones?, Is it Scalable ?")
#query("What is Azure Platform Landing Zone?")
#query("What are Azure Functions used for?")