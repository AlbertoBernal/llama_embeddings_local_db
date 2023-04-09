from langchain.vectorstores import Chroma

from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader

GPT4ALL_MODEL_PATH = "/home/alberto/Downloads/LLM_MODELS/gpt4all-lora-quantized-new.bin"
VICUNA_7B_MODEL_PATH= "/home/alberto/Downloads/LLM_MODELS/ggml-vicuna-7b-4bit-rev1.bin"
llama_embeddings = LlamaCppEmbeddings(model_path=VICUNA_7B_MODEL_PATH)
#llm = LlamaCpp(model_path=VICUNA_7B_MODEL_PATH)

# Load and process the text
loader = TextLoader('state_of_the_union.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)



# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = 'db_2'

vectordb = Chroma.from_documents(documents=texts, embedding=llama_embeddings, persist_directory=persist_directory)

