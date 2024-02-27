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

load_dotenv()

# set_debug(True)

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

template = """Answer the question based only on the following context:
{context}

Question: {question}"""

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Combine relevant documents into the context for the requested question
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Pass the question into the retriever to return similar results
retrieved_docs = RunnableParallel({
    "docs": itemgetter("question") | retriever ,
    "question": lambda x: x["question"]
})

# take all the docs from the prior step (which includes the question --> retriever --> relevant docs)

context_builder = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
answer = {
    "answer": context_builder | ANSWER_PROMPT | model,
    "docs": itemgetter("docs")
}

chain = (retrieved_docs | answer)
result = chain.invoke({"question":passed_query})

####### Formatted output
print(f"\n❓ '{passed_query}'", end="\n\n")
print("*" * 80)
print("💡 " + result['answer'].content)

print("\n" + "- " * 2, end="")
print(f"Supporing Docs", end="")
print("- " * 30)

for (idx, doc) in enumerate(result["docs"]):
    print(f"\t🥝 {doc.metadata['title']} 🔗 {doc.metadata['link']}")

print("*" * 80)
