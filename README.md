# Learning RAG with my Wordpress Data

This small project aims to incorporate your WordPress data into a knowledge base that utilizes Retrieval-Augmented Generation (RAG) techniques.

# Getting Started

```
pip install python-dotenv requests pandas langchain langchain_community html2text sentence_transformers
```

Create a `.env` file using `env.example` as a guide. This will be important for downloading your wordpress data. Make sure to copy your Wordpress username, application password, and blog domain into the `.env`. NOTE: This assume the blog is at the root path.

Download the wordpress content to the `./data` directory:

```
python3 wordpress-dl.py
```

# PostgreSQL / PGVector

For pgvector embedding support:

```
pip install  pgvector psycopg2-binary
```

Make sure you have the proper `.env` values. While the defaults are fine for local development, there's a chance you might have something different if using an existing Postgres database (i.e. not running the docker image)

Run the docker image in `./postgres`

```
docker compose up
```

# Run Embeddings

Run the embeddings script to take the wordpress download and save embeddings to the postgres database. This may take some time. _If you want to insert data into the database, that is, wipe and start with clean data, pass the `--embed` CLI argument:

```
python3 embed.py --embed
```

By default only 1 record is inserted for testing (ie by not specifying `--limit` and then some integer > 1, it will only create the vector embedding for the first record) - so you can test easier without spending too much extra compute time.

If you are curious about additional information, pass `--verbose`

A sample invocation could look like:

```
python3 embed.py --verbose --embed --limit 100
```

...but increase your limit appropriately sized for your wordpress blog. No problem if the number is larger than the number of entries - it's smart enough to import up to the max record count.

