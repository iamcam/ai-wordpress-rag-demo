# Start here as a guide, though we're not following it
# https://python.langchain.com/docs/expression_language/cookbook/retrieval
import time

script_start = time.time()


import argparse
import os

# There's a warning/error thrown, this will resolve it
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from operator import itemgetter

from dotenv import load_dotenv
from langchain.globals import set_debug, set_verbose
from langchain.prompts.prompt import PromptTemplate
from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, format_document
from langchain_core.runnables import RunnableParallel

from utils import PG_CONNECTION_STRING, embeddings

load_dotenv()
# set_debug(True)

####################################
# Set up cli args
####################################

parser = argparse.ArgumentParser(description='Information retrieval from blog archives')
parser.add_argument('--collection',  default=os.environ.get("PGVECTOR_COLLECTION_NAME"), type=str, help='Collection name to use in the DB store. Only change if you need something specific', required=False)
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output - provide more output than usual. May be helpful.', required=False)
parser.add_argument('--use-api', action='store_true', help='Use OpenAI/ChatGPT APIs instead of local inference. An OpenAI API key is required (see README for details)', required=False)
parser.add_argument('--query', '-q', default="What is RaspberryPi?", type=str, help='Curious minds want to know...?', required=False)

args = parser.parse_args()

VERBOSE = args.verbose
if VERBOSE:
    print(f"{args}\n")

set_verbose(VERBOSE) # langchain verbosity

db_collection = args.collection
passed_query = args.query
USE_API = args.use_api

####################################
# Conditional Imports
####################################
if USE_API:
    from langchain_openai import ChatOpenAI

else:
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import \
        StreamingStdOutCallbackHandler
    from langchain_community.llms.llamacpp import LlamaCpp

####################################
# Vector Store + Retriever
####################################

store = PGVector(
    collection_name=db_collection,
    connection_string=PG_CONNECTION_STRING,
    embedding_function=embeddings,
)
retriever = store.as_retriever()

####################################
# Prompting
####################################

# Primes the LM and gives the context via the documents returned from the retriever
system_instruction = """Answer the question based only on the following context:
{context}
"""

# Simply user instruction
user_instruction = "Question: {question}"


if USE_API:
    print("🤖 Using OpenAI API")
    model = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.75)

    # OpenAI templates are straightforward, system instruction + Question
    template = f"""{system_instruction}

    {user_instruction}"""

else:
    print("🤖 Using local inference")

    model = LlamaCpp(
        model_path="./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        verbose=VERBOSE,  # Verbose is required to pass to the callback manager
    )

    # Llama uses a bit more structured query templates, but otherwise the relevant information is the same as OpenAI's above
    template = f"""<|im_start|>system
    {system_instruction}
    <|im_end|>
    <|im_start|>user
    {user_instruction}<|im_end|>
    <|im_start|>assistant
    """

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Combine relevant documents into the context for the requested question
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    # ideally this uses actual tokens. OpenAI seems to be able to take the full text,
    # but TinyLlama needs shorter - max context size is 512, but the max chars is likely
    # several times larger. Look into `LlamaTokenizer` from the HF transformers package
    # (outside the scope of this demonstration, IMO).
    if USE_API:
        part = None
    else:
        maxchars = 1200 #guestimage ~3chars/token avg, leaving some room for query text etc
        part = int(maxchars/len(docs))

    doc_strings = []
    for doc in docs:
        formatted_doc = format_document(doc, document_prompt)[:part]
        doc_strings.append(formatted_doc)
    return document_separator.join(doc_strings)

# Pass the question into the retriever to return similar results
retrieved_docs = RunnableParallel({
    "docs": itemgetter("question") | retriever ,
    "question": lambda x: x["question"]
})

# take all the docs from the prior step (which includes the question --> retriever --> relevant docs)

context_builder = {
    "question": itemgetter("question"),
    "context": lambda x: _combine_documents(x["docs"]),
}

answer = {
    "answer": context_builder | ANSWER_PROMPT | model | StrOutputParser(),
    "docs": itemgetter("docs")
}

chain = (retrieved_docs | answer)
result = chain.invoke({"question":passed_query})

if VERBOSE:
    print(f"RESULT: {result}")

####################################
# Formatted Output
####################################

print("*" * 80)

print(f"\n❓ '{passed_query}'", end="\n\n")

print("- " * 40)

if hasattr(result['answer'], "content"):
    response_anser = result['answer'].content
else:
    response_anser = result['answer']

print("💡 " + response_anser)

print("\n" + "- " * 2, end="")
print(f"Supporing Docs", end=" ")
print("- " * 30)

for (idx, doc) in enumerate(result["docs"]):
    print(f"\t🥝 {doc.metadata['title']} 🔗 {doc.metadata['link']}")

print("\n" + "~ "* 5 + f"Finished in {time.time() - script_start :.2f}s " + "~ "* 10 + "\n")

print("*" * 80)