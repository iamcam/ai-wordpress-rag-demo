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

Install required for OpenAI API usage

```sh
pip install langchain_openai
```

# Using Data

Documents (embedded and retrieved, btw) have the following general structure in this project.

```
# document
    # page_content
    # metadata
        # id
        # link
        # title
        # categories
        # tags
```

Results are often returned as a `List` of tuples (idx, Document), so it's appropriate to enumerate over the list:

```python
for (idx, doc) in enumerate(results["docs]):
    print(f"{doc.metadata['title']}")
```
Most useful data for augmenting the LLM responses will be included in the `metadata` property, a dictionary of data fed in during embedding.

# Query

Running queries from the CLI is simple:

```sh
python3 query.py --query "How do you use swift to make LEDS blink on a raspberry pi?"
```

After a few moments, expect to see a response like this,

```

❓ 'How do you use swift to make LEDS blink on a raspberry pi?'

********************************************************************************
💡 To make LEDs blink on a Raspberry Pi using Swift, you need to first set up your Raspberry Pi with Swift properly installed. You can follow the instructions provided by Joe Bell on his website to install Swift 2.2 via apt-get. Once you have Swift installed, you can use a library like SwiftyGPIO to interact with the GPIO pins on the Raspberry Pi.

With SwiftyGPIO, you can define the GPIO pins you want to use, set their direction (IN for sensors, OUT for LEDs), and control their values to turn the LEDs on and off. By writing a simple Swift program that toggles the GPIO pin values in a loop, you can make the LEDs blink.

Additionally, you can also incorporate other components like temperature sensors (DS18B20) into your project to further enhance the functionality of your Raspberry Pi project. By following the wiring diagrams and using Swift to read sensor data and control GPIO pins, you can create more complex projects involving LEDs and sensors on the Raspberry Pi.
- - - - - - - - - - - - - - - Supporing Docs- - - - - - - - - - - - - - -
        🥝 Currently, Swift on Raspberry Pi3 🔗 https://archive.mistercameron.com/2016/04/currently-swift-on-raspberry-pi3/
        🥝 Accessing Raspberry Pi GPIO Pins (With Swift) 🔗 https://archive.mistercameron.com/2016/06/accessing-raspberry-pi-gpio-pins-with-swift/
        🥝 Ready Your Raspberry Pi for Swift 🔗 https://archive.mistercameron.com/2016/05/ready-your-raspberry-pi-for-swift/
        🥝 Compile Swift 3.0 on Your ARM computers (Raspberry Pi, BeagleBone Black, etc) 🔗 https://archive.mistercameron.com/2016/06/compile-swift-3-0-on-your-arm-computer/
********************************************************************************
```