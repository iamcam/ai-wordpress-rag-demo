import time
import warnings

warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores.pgvector import PGVector

from utils import embeddings  # HuggingFaceBgeEmbeddings
from utils import PG_CONNECTION_STRING, chunk_list

load_dotenv()

parser = argparse.ArgumentParser(description='Information retrieval from blog archives')
parser.add_argument('--posts-json', type=str, help='Path to posts json file', required=False)
parser.add_argument('--collection', default=os.environ.get("PGVECTOR_COLLECTION_NAME", "web_content"), type=str, help='Collection name to use in the DB store. Only change if you need something specific', required=False)
parser.add_argument('--limit', type=int, help='Limit to the first N blog posts to embed. Useful for testing', required=False)
parser.add_argument('--embed', action='store_true', help='Perform embedding, wiping all previous embedding from the DB and starting fresh', required=False)
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output - provide more output than usual. May be helpful.', required=False)

args = parser.parse_args()

RECORD_LIMIT = args.limit

print("-" * 80)
print(f"Limiting to {RECORD_LIMIT} records. (Default 1 for your own testing sanity)")
print("-" * 80)

# Run embedding - will reset the database with fresh embeddings
EMBED = args.embed == True

VERBOSE = args.verbose
if VERBOSE:
    print(f"{args}\n")


posts_path = args.posts_json
if posts_path == "":
    print("Please specify a posts json data with the --posts argument")
    exit(1)
elif args.posts_json is None:
    posts_path = "data/posts.json"

db_collection = args.collection

## Load posts data into pandas
df = pd.read_json(posts_path)

def extract_rendered_title(row):
    return row['title']['rendered']
df['title'] = df.apply(extract_rendered_title, axis=1)

def extract_rendered_content(row):
    return row['content']['rendered']
df['content'] = df.apply(extract_rendered_content, axis=1)


## Load Pandas DataFrame into langchain loader
post_loader = DataFrameLoader(df, page_content_column='content')
docs = post_loader.load()[:RECORD_LIMIT]

## transform any html to text
html2text = Html2TextTransformer()
docs_transformed_html = html2text.transform_documents(docs)

if VERBOSE:
    print(docs_transformed_html[:1])

## Split into smaller chunks

# Recommended for generic text, splitting in order until chunks are small enough
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=250,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )
# split_docs = text_splitter.transform_documents(docs_transformed_html)

# if VERBOSE:
#     print("First 3 docs")
#     print(split_docs[:3])

## Embedding

# Using the full document rather than partial, as the returned results are helpful as full articles
docs_to_embed = docs_transformed_html[:RECORD_LIMIT]

# embeddings are imported, used across scripts

connection_string = PG_CONNECTION_STRING

# The name of the collection to use. (default: langchain) NOTE: This is not the name of the table, but the name of the collection. The tables will be created when initializing the store (if not exists) So, make sure the user has the right permissions to create tables.

if VERBOSE:
    print(f"Creating connection to database with collection_name={db_collection}")

if EMBED:
    record_count = len(docs_to_embed)

    print(f"Importing {record_count} records")
    if VERBOSE:
        start_time = time.time()
        print(f"Begin embeddings and storage.")
        print(f"Please be patient. This may take some time...")

    # due to memory constraints, may need to reduce the concurrent embeddings

    #https://api.python.langchain.com/en/stable/vectorstores/langchain_community.vectorstores.pgvector.PGVector.html#langchain_community.vectorstores.pgvector.PGVector.create_tables_if_not_exists
    db = PGVector.from_documents(
        embedding=embeddings,
        documents = [], # Will load in batches below for status
        collection_name=db_collection,
        connection_string=connection_string,
        pre_delete_collection=EMBED,
    )


    # number of docs to chunk at a time. helps with memory overhead and constrained systems (like a laptop),
    # otherwise you'll get an out-of-memory error
    chunk_size = 10

    ###############################################
    ## Progress Bar /////////////////]
    # the next several print statement will print out a progress bar, FYI
    if VERBOSE:
        print(f"Doc Size: {len(docs_to_embed)}\tChunk Size: {chunk_size}")

    print("/] 0%", end='', flush=True)
    for idx, chunk in chunk_list(docs_to_embed, chunk_size):
        db.add_documents(chunk)
        print("\r" + "/" * max(1,int(idx / record_count * 100)), end='', flush=True)
        print(f"] {int(idx / record_count * 100)}%", end="", flush=True)
    print("\r" + "/" * int(100), end='', flush=True)
    print(f"] 100%", end='', flush=True)

    if VERBOSE:
        end_time = time.time()
        print("")
        print(f"Encoding finished, {end_time - start_time}s elapsed")

else:
    db = PGVector(
        collection_name=db_collection,
        connection_string=connection_string,
        embedding_function=embeddings,
    )

# store.add_documents(split_docs[0:5])
similarity_search_term = "Accessing Raspberry Pi"
print(f"\n🍋 Searching similarity with 👉 '{similarity_search_term}'")

if VERBOSE:
    start_time = time.time()
    print(f"\nStarting similarity search.")

docs_with_score = db.similarity_search_with_score(similarity_search_term)

if VERBOSE:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Search finished: {elapsed:.2f}s elapsed ({elapsed/record_count:.2f} sec/record)\n")
    print("Results:")
    for doc, score in docs_with_score[:3]:
        print("-" * 80)
        print("Score: ", score)
        print(doc.metadata['title'])
        print("-" * 80)

# TODO: Consider adding index indexes (HNSW, IVFFLAT)
# print(f"Table name: {db.CollectionStore.__tablename__}")