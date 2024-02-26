# Learning RAG with my Wordpress Data

This small project aims to incorporate your WordPress data into a knowledge base that utilizes Retrieval-Augmented Generation (RAG) techniques, using only free and open-source libraries and models.

The premise is simple - a series of scripts will perform specific actions. By doing this, the "what is happening" is self-contained to each script, which helps when you're still learning the process.

This example should be able to run on on CPU if you don't have a compatible GPU, but YMMV.

# Getting Started

This is a fairly easy project to get up and running...

Create a `.env` file using `env.example` as a guide.

```sh
cp env.example .env
```

This will be important for downloading your wordpress data. Make sure to copy your Wordpress username, application password, and blog domain into the `.env`. NOTE: This assume the blog is at the root path.

```sh
pip install --upgrade python-dotenv requests pandas langchain langchain_community html2text sentence_transformers
```

Download the wordpress content to the `./data` directory (this assumes your blog is at the domain root, for now):

```sh
python3 wordpress-dl.py
```

# PostgreSQL / PGVector

For pgvector embedding support:

```sh
pip install pgvector psycopg2-binary
```

Make sure you have the proper `.env` values. While the defaults are fine for local development, there's a chance you might have something different if using an existing Postgres database (i.e. not running the docker image)

Run the docker image in `./postgres`

```sh
docker compose up
```

# Run Embeddings

Run the embeddings script to take the wordpress download and save embeddings to the postgres database. This may take some time. _If you want to insert data into the database, that is, wipe and start with clean data, pass the `--embed` CLI argument:

```sh
python3 embed.py --embed
```

By default only 1 record is inserted for testing (ie by not specifying `--limit` and then some integer > 1, it will only create the vector embedding for the first record) - so you can test easier without spending too much extra compute time.

If you are curious about additional information, pass `--verbose`

A sample invocation could look like:

```sh
python3 embed.py --verbose --embed --limit 100
```

...but increase your limit appropriately sized for your wordpress blog. No problem if the number is larger than the number of entries - it's smart enough to import up to the max record count.

# Query Your Data

Install required

```sh
pip install langchain_openai
```
