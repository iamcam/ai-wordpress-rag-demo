# RAG with Wordpress Data - Demo

This small project aims to incorporate WordPress blog entries into a knowledge base that utilizes Retrieval-Augmented Generation (RAG) techniques.

The premise of this project is simple - a series of scripts will perform specific actions. Each script is self-contained, which helps demonstrate each action within its own context, relatively simply.

By default, the included scripts use free and open-source libraries and models, though an option to use OpenAI's LLMs is available if you have access. This example should be able to run on on CPU if you don't have a compatible GPU, but YMMV.

## RAG in a Nutshell

RAG makes use of two components: a retrieval system and a generative language model. The retrieval system queries documents relevant to a query from a data store / knowledge base (for example, a collection of blog posts). The retrieved documents are then fed into a language model, which produces informed responses for the user. One of the primary benefits of using RAG is that it allows the AI system to extend beyond the training data without the need for fine-tuning or re-training - data can be added and updated in the data store dynamically. Query results can include not just he language model-supplied text, but also a collection of relevant documents or sources. You can read more about RAG at the [IBM Research Blog](https://research.ibm.com/blog/retrieval-augmented-generation-RAG).

This is a fairly easy project to get up and running...

## Setup

First set up the project's virtual environment and activate it

```sh
python3 -m venv virtualenv
source ./virtualenv/bin/activate
```

You may install all the dependencies for this project by using the `requirements.txt` file. If you do, there is no need to manually install other packages via `pip` in this document.

```sh
pip install -r requirements.txt
```
## Environment

Create a `.env` file using `env.example` as a guide.

```sh
cp env.example .env
```

This will be important for downloading your wordpress data. Make sure to copy your Wordpress username, application password, and blog domain into the `.env`. NOTE: This assume the blog is at the root path.

## Loading Wordpress Data

```sh
pip install --upgrade python-dotenv requests pandas langchain langchain_community html2text sentence_transformers
```

Download the wordpress content to the `./data` directory (this assumes your blog is at the domain root, for now):

```sh
python3 wordpress-dl.py
```

## PostgreSQL / PGVector

PGVector is used for storing and query text embeddings (vectors) in a Postgresql database. One benefit of PGVector is that it can be added to existing database systems and does not require proprietary third-party products.

For pgvector embedding support:

```sh
pip install pgvector psycopg2-binary
```

Make sure you have the proper `.env` values. While the defaults are fine for local development, there's a chance you might have something different if using an existing Postgres database (i.e. not running the docker image). Here, we make use of a Docker image to keep setup simple, but you could easily use another PostgreSQL instance if you are able to add and load the pgvector extension yourself.

To run the docker image in `./postgres`:

```sh
docker compose up
```

## Run Embeddings

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

## Query Your Data

This repo demonstrates both local and api-based use cases. One of the benefits of a locally-run model is that your data stays private and will not be used for training others' models. Furthermore, there can be some performance gains by keeping queries local to a server or network, not does it does not incur API usage costs on self-hosted systems. On the other hand, API usage may be desired, as OpenAI has much larger context windows than the model used in this project, and the OpenAI models can be quite good.

### Using HuggingFace Local Pipelines (Default)

Local language model usage will need the `trsnformers` python package and pytorch (`torch`)

```sh
pip install transformers torch llama-cpp-python
```

Next, download the model used in this offline approach. Because language models can be large (several GB), I chose a smaller Llama model that demonstrates. Even smaller models can be used, but oftentimes at the limit of context windows, which can be alleviated, but outside the scope of this project. This project uses `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (`Q4_K_M`)

(If you don't have `huggingface-cli` installed, you can find [details here](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)).

```sh
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ./models/ --local-dir-use-symlinks True
```

This command will download the model to the user account's cache director, and will symlink from the models directory.

Once the model is downloaded, you may run local inference, which is the default option. See the next section, **Query** for instructions.

### Using OpenAI API

Install required for OpenAI API usage

```sh
pip install langchain_openai
```

Ensure you have your OpenAI API key saved to your `.env` file. You can configure it in the [OpenAI Platform API Key tab](https://platform.openai.com/api-keys): `OPENAI_API_KEY="...`

### Using LLM and Document Data

Documents (embedded and retrieved, btw) have the following general structure in this project.

```
## document
    ## page_content
    ## metadata
        ## id
        ## link
        ## title
        ## categories
        ## tags
```

Results are often returned as a `List` of tuples (idx, Document), so it's appropriate to enumerate over the list:

```python
for (idx, doc) in enumerate(results["docs]):
    print(f"{doc.metadata['title']}")
```
Most useful data for augmenting the LLM responses will be included in the `metadata` property, a dictionary of data fed in during embedding.

## Query

Running queries from the CLI is simple..

...local model:

```sh
python3 query.py --query "Does RaspberryPi have GPIO that swift can use?"
```

...using OpenAI:

```sh
python3 query.py --query "Does RaspberryPi have GPIO that swift can use?" --use-api
```

After a few moments, expect to see a response like this,

```
❓ 'Does RaspberryPi have GPIO that swift can use?'

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
💡 Yes, RasbperryPi has GPIO that swift can use as per the context given.

- - Supporing Docs - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        🥝 Accessing Raspberry Pi GPIO Pins (With Swift) 🔗 https://archive.mistercameron.com/2016/06/accessing-raspberry-pi-gpio-pins-with-swift/
        🥝 Currently, Swift on Raspberry Pi3 🔗 https://archive.mistercameron.com/2016/04/currently-swift-on-raspberry-pi3/
        🥝 Compile Swift 3.0 on Your ARM computers (Raspberry Pi, BeagleBone Black, etc) 🔗 https://archive.mistercameron.com/2016/06/compile-swift-3-0-on-your-arm-computer/
        🥝 Ready Your Raspberry Pi for Swift 🔗 https://archive.mistercameron.com/2016/05/ready-your-raspberry-pi-for-swift/

~ ~ ~ ~ ~ Finished in 14.80s ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

********************************************************************************
```

There are additional arguments that can be helpful when testing your data and models. run `python3 query.py --help` for more options.
