# This is installing the pgvector extension for postgres
FROM postgres:16.2

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

FROM pgvector/pgvector:pg16

