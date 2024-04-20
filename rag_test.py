from dotenv import load_dotenv

load_dotenv()


# Load files of many types

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('resources/', exclude=["*.mp4", "*.json"])
raw_docs = loader.load()
print("raw docs", raw_docs)


# Split documents into small chunks. This allows us to identify the most
# relevant segments for a query and feed only those into the LLM

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(raw_docs)
print("\nsplit 0", docs[0])
print("split 1", docs[1])


# Generate embeddings for each segment and integrate them into the Chroma
# vector database

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(),
                                    persist_directory="./chroma_db")

query = "Que apredizados devo apresentar ao fim dessa unidade de HTML?"
search_docs = vectorstore.similarity_search(query)

print("\nquery", query)
print("related content", search_docs)


# Indexing: load and keep in sync documents from any source into a vector store

from langchain.indexes import SQLRecordManager, index

namespace = f"chroma/index"
record_manager = SQLRecordManager(namespace,
                                  db_url="sqlite:///record_manager_cache.sql")

record_manager.create_schema()

print('')
print(index(
    docs,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key="source"))

removed_doc = docs.pop()
print("removing doc")

print(index(
    docs,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key="source"))

docs.append(removed_doc)
print("append doc")

print(index(
    docs,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key="source"))
