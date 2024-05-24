from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

documents = SimpleDirectoryReader("data").load_data()


llm = Ollama(model="llama3", request_timeout=360.0, temperature=0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
context_window = 4096
num_output = 300
chunk_overlap_ratio = 0.2

chunk_size_limit = 600

prompt_helper = PromptHelper(
    context_window=context_window,
    num_output=num_output,
    chunk_overlap_ratio=chunk_overlap_ratio, 
    chunk_size_limit=chunk_size_limit
)

Settings.embed_model = embed_model
Settings.llm = llm
Settings.prompt_helper = prompt_helper

index = VectorStoreIndex.from_documents(
    documents=documents,
)

query_engine = index.as_query_engine(streaming=True)

prompt_suffix = ""
prompt_suffix += "\nRespond in Vietnamese"
prompt_suffix += "\nWrite a lengthy response to this query"


response = query_engine.query("quy trình 5s là gì" + prompt_suffix)
# response = query_engine.query("Quy trình 5S dành cho ai" + prompt_suffix)
# print(response)

response.print_response_stream()
