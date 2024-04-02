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
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.documents import Document
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.language_models.llms import LLM

from utils import PG_CONNECTION_STRING, embeddings, dedupe_docs, fetch_document

load_dotenv()
# set_debug(True)

####################################
# Set up cli args
####################################

parser = argparse.ArgumentParser(description='Information retrieval from blog archives')
parser.add_argument('--collection',  default=os.environ.get("PGVECTOR_COLLECTION_NAME"), type=str, help='Collection name to use in the DB store. Only change if you need something specific', required=False)
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output - provide more output than usual. May be helpful.', required=False)
parser.add_argument('--use-api', action='store_true', help='Use OpenAI/ChatGPT APIs instead of local inference. An OpenAI API key is required (see README for details)', required=False)
parser.add_argument('--temperature', '-t', default=0.75, type=float, help='Set the temperature for the model. Values typically range from 0 (more consistent/less creative) to 1 (more diverse and creative)', required=False)
parser.add_argument('--query', '-q', default="What is RaspberryPi?", type=str, help='Curious minds want to know...?', required=False)

args = parser.parse_args()

VERBOSE = args.verbose
if VERBOSE:
    print(f"{args}\n")

set_verbose(VERBOSE) # langchain verbosity

db_collection = args.collection
passed_query = args.query
USE_API = args.use_api
use_temp = args.temperature

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
MAX_TOKENS = 4096
CONTEXT_WINDOW = 512

# Primes the LM and gives the context via the documents returned from the retriever
system_instruction = """Answer the question based only on the following context:
{context}
"""

# Simply user instruction
user_instruction = "Question: {question}"

## Llama 2 is 4096, according to docs, OpenAI/GPT-4 is much more

if USE_API:
    print("🤖 Using OpenAI API")
    model = ChatOpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=use_temp)
    # OpenAI templates are straightforward, system instruction + Question
    template = f"""{system_instruction}

    {user_instruction}"""

else:
    print("\n🤖 Using local inference")

    model = LlamaCpp(
        model_path="./models/llama-2-7b-chat.Q4_K_M.gguf",
        temperature=use_temp,
        model_kwargs={
            "context_window":MAX_TOKENS,
            "max_new_tokens":20,
        },
        max_tokens=MAX_TOKENS,
        # top_p=1,
        verbose=VERBOSE,  # Verbose is required to pass to the callback manager
    )

    # Llama uses a bit more structured query templates, but otherwise the relevant information is the same as OpenAI's above
    template = f"""<s>[INST] <<SYS>>
    {system_instruction}
    <</SYS>>
    {user_instruction} [/INST]
    """

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

token_counts = {"documents": None, "query": None, "response": None}

# Combine relevant documents into the context for the requested question
# i
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
) -> str:
    doc_strings = []

    for doc in docs:
        if False: #full documents overwhelm the LM
            doc_body = fetch_document("data/posts.json", doc.metadata['id'])
            full_doc = Document(page_content=doc_body, metadata=doc.metadata)
            html2text = Html2TextTransformer()
            transformed = html2text.transform_documents([full_doc])

            formatted_doc = format_document(transformed[0], document_prompt)
        else:
            formatted_doc = format_document(doc, document_prompt)
        doc_strings.append(formatted_doc)

    joined_docs = document_separator.join(doc_strings)

    if model.get_num_tokens:
        num_tokens = model.get_num_tokens(joined_docs)
        if VERBOSE: print(f"🔆🔆 Num Tokens in docs: ({num_tokens})")

        while num_tokens > CONTEXT_WINDOW-25:
            joined_docs = joined_docs[5:]
            num_tokens = model.get_num_tokens(joined_docs)
        token_counts["documents"] = num_tokens
    else:
        # severely limits the text, but in absence of token counting, be safe
        joined_docs = joined_docs[:CONTEXT_WINDOW]

    return joined_docs

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
    "docs": lambda x: x["docs"]
}

chain = (retrieved_docs | answer)

if VERBOSE:
    chain.get_graph().print_ascii()

print("Processing.....")

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
    response_answer = result['answer'].content
else:
    response_answer = result['answer']

if model.get_num_tokens:
    token_counts["response"] = model.get_num_tokens(response_answer)
    token_counts['query'] = model.get_num_tokens(passed_query)

print(f"\t🫧  Query\t({token_counts['query']} tokens)")
print(f"\t🫧  Documents\t({token_counts['documents']} tokens)")
print(f"\t🫧  Response\t({token_counts['response']} tokens)")
print(f"\t🫧  Total\t({token_counts['documents'] + token_counts['response'] + token_counts['query']} tokens)")
print("\n")
print(f"💡 {response_answer}")

print("\n" + "- " * 2, end="")
print(f"Supporing Docs", end=" ")
print("- " * 30, end="\n\n")

for (idx, doc) in enumerate(dedupe_docs(result["docs"])):
    print(f"\t🥝 {doc.metadata['title']} 🔗 {doc.metadata['link']}")

print("\n" + "~ "* 5 + f"Finished in {time.time() - script_start :.2f}s " + "~ "* 10 + "\n")

print("*" * 80)