import pandas as pd


# Puls a given blog entry from the data at the specified path and matching a given ID
def fetch_document(path: str, id: int) -> str:
    df = pd.read_json(path)
    doc = df.query(f"id == {id}").iloc[0]
    body = doc.content['rendered']
    return body

