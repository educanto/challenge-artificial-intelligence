import os
import dotenv

# from langchain_community.document_loaders import TextLoader
#
# loader = TextLoader("resources/Apresentação.txt")
# loader.load()

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('resources/', exclude=["*.mp4", "*.json"])
docs = loader.load()
print(docs)

