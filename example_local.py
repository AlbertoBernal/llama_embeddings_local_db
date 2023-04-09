from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import VectorDBQA, RetrievalQA

VICUNA_7B_MODEL_PATH= "/home/alberto/Downloads/LLM_MODELS/ggml-vicuna-7b-4bit-rev1.bin"
persist_directory = 'db_2'
llama_embeddings = LlamaCppEmbeddings(model_path=VICUNA_7B_MODEL_PATH, n_batch=512)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=llama_embeddings)
llm = LlamaCpp(model_path=VICUNA_7B_MODEL_PATH, max_tokens=128)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={"k": 1}))

query = """
Identify three things the president said about Ketanji Brown Jackson. Provide the answer in the form: 

- ITEM 1
- ITEM 2
- ITEM 3
"""
output=qa.run(query)

print ("RESULT:")
print (output)
