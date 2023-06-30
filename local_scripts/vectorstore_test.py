# -*- coding: utf-8 -*-
# @Time : 5/22/23 7:24 PM
# @Author : AndresHG
# @File : vectorstore_test.py
# @Email: andresherranz999@gmail.com
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma


embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"}
    )
vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

if __name__ == "__main__":
    query = "Which class should I pick if I like spells?"

    result = vectorstore.similarity_search(query)
    for i, doc in enumerate(result):
        print(f"Document {i}:")
        print(doc)

