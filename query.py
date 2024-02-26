# Start here as a guide, though we're not following it
# https://python.langchain.com/docs/expression_language/cookbook/retrieval
import argparse
import os

# There's a warning/error thrown, this will resolve it
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils import PG_CONNECTION_STRING, embeddings

set_debug(True)
set_verbose(True)

load_dotenv()

####################################
# Set up cli args
####################################

parser = argparse.ArgumentParser(description='Information retrieval from blog archives')
parser.add_argument('--collection',  default=os.environ.get("PGVECTOR_COLLECTION_NAME"), type=str, help='Collection name to use in the DB store. Only change if you need something specific', required=False)
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output - provide more output than usual. May be helpful.', required=False)

parser.add_argument('--query', '-q', default="What is RaspberryPi?", type=str, help='Curious minds want to know...?', required=False)

args = parser.parse_args()

VERBOSE = args.verbose
if VERBOSE:
    print(f"{args}\n")

db_collection = args.collection
passed_query = args.query

####################################
# Set up cli args
####################################

# Instantiate Vector Store

#vector_store and retriever
store = PGVector(
    collection_name=db_collection,
    connection_string=PG_CONNECTION_STRING,
    embedding_function=embeddings,
)
retriever = store.as_retriever()

# Prompt Template for the LM
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# model creation - https://python.langchain.com/docs/integrations/llms/huggingface_pipelines#model-loading
# model_name = "Llama-2-13B-chat-GGML" # HuggingFace/TheBloke
# model_name = "gpt2"

model = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Chain - No docs returned, just answers
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

response_dict = chain.invoke(passed_query)