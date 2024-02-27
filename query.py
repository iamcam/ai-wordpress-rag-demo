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

# set_debug(True)


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

set_verbose(VERBOSE) # langchain verbosity

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

#####################################################################
# Prompting

## Document Prompt, for inclusion
from operator import itemgetter

from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in US English.

Follow Up Input: {question}
"""

template = """Answer the question based only on the following context:
{context}

Question: {question}"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Combine relevant documents into the context for the requested question
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# The question alone
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}

retrieved_docs = {
    "docs": itemgetter("standalone_question") | retriever ,
    "question": lambda x: x["standalone_question"]
}

final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs")
}
# (Standalone) Question...
#   ==> Add context of retrieved docs from retriever
#   ==> Combine
final_chain = RunnablePassthrough() | standalone_question | retrieved_docs | answer

inputs = {"question": passed_query}
result = final_chain.invoke(inputs)

print(f"\n❓ '{passed_query}'", end="\n\n")
print("*" * 80)
print("💡 " + result['answer'].content)

print("- " * 15, end="")
print(f"Supporing Docs", end="")
print("- " * 15)

for (idx, doc) in enumerate(result["docs"]):
    print(f"\t🥝 {doc.metadata['title']} 🔗 {doc.metadata['link']}")

print("*" * 80)