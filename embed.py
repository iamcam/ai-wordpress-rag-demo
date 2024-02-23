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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()

parser = argparse.ArgumentParser(description='Information retrieval from blog archives')
parser.add_argument('--posts-json', type=str, help='Path to posts json file', required=False)
parser.add_argument('--limit', default=1, type=int, help='limit to first N posts, choose between', required=False)
parser.add_argument('--embed', action='store_true', help='Perform embedding, wiping all previous embedding from the DB and starting fresh', required=False)
parser.add_argument('--verbose', action='store_true', help='Verbose output - provide more output than usual. May be helpful.', required=False)

args = parser.parse_args()

if args.limit < 1:
    print("Please specify a post limit of 1 or more")
    exit(1)
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

## Load posts data into pandas
df = pd.read_json(posts_path)

desired_xform = df.iloc[0].content['rendered']
def extract_rendered_title(row):
    return row['title']['rendered']
df['title'] = df.apply(extract_rendered_title, axis=1)

def extract_rendered_content(row):
    return row['content']['rendered']
df['content'] = df.apply(extract_rendered_content, axis=1)


## Load Pandas DataFrame into langchain loader
post_loader = DataFrameLoader(df, page_content_column='content')
docs = post_loader.load()

## transform any html to text
html2text = Html2TextTransformer()
docs_transformed_html = html2text.transform_documents(docs)

## Split into smaller chunks

# Recommended for generic text, splitting in order until chunks are small enough
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
split_docs = text_splitter.transform_documents(docs_transformed_html)

## Perform embedding (local-only)
embeddings = HuggingFaceBgeEmbeddings()

# Manual embedding - not needed if using the langchain PGVector connector
# texts = [doc.page_content for doc in split_docs]
# doc_embeddings = embeddings.embed_documents(texts[:3])

## Embedding

from langchain_community.vectorstores.pgvector import PGVector

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "blogvector"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
)



# The name of the collection to use. (default: langchain) NOTE: This is not the name of the table, but the name of the collection. The tables will be created when initializing the store (if not exists) So, make sure the user has the right permissions to create tables.
COLLECTION_NAME = "web_content"
if VERBOSE:
    print(f"Creating connection to database with collection_name={COLLECTION_NAME}")

if EMBED:
    if VERBOSE:
        start_time = time.time()
        print(f"Begin embeddings and storage.")
        print(f"Please be patient. This may take some time")

    #https://api.python.langchain.com/en/stable/vectorstores/langchain_community.vectorstores.pgvector.PGVector.html#langchain_community.vectorstores.pgvector.PGVector.create_tables_if_not_exists
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=docs[:(min(RECORD_LIMIT, len(docs)))],
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=EMBED,
    )

    if VERBOSE:
        end_time = time.time()
        print(f"Encoding finished, {end_time - start_time}s elapsed\n")
else:
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
    )

# store.add_documents(split_docs[0:5])
similarity_search_term = "Accessing Raspberry Pi"
if VERBOSE:
    start_time = time.time()
    print(f"\nStarting similarity search.")
print(f"Searching similarity with '{similarity_search_term}'")

docs_with_score = db.similarity_search_with_score(similarity_search_term)
if VERBOSE:
    end_time = time.time()
    print(f"Search finished: {end_time - start_time}s elapsed\n")

if VERBOSE:
    print("Results:")
    for doc, score in docs_with_score[:3]:
        print("-" * 80)
        print("Score: ", score)
        print(doc.metadata['title'])
        print("-" * 80)

# TODO: Consider adding index indexes (HNSW, IVFFLAT)
print(db.CollectionStore.__tablename__)