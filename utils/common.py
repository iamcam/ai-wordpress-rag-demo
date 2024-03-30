from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def dedupe_docs(docs: [Document]):
    unique_texts = {}
    skipped = 0
    deduped_docs = []
    for doc in docs:
        if doc.metadata['id'] not in unique_texts:
            unique_texts[doc.metadata['id']] = True
            deduped_docs.append(doc)
        else:
            skipped += 1
    return deduped_docs