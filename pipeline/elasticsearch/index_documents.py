from elasticsearch import Elasticsearch, helpers
import pandas as pd

def index_documents(es_url, index_name, data):
    """
    Bulk index documents into Elasticsearch.

    Args:
        es_url (str): Elasticsearch URL.
        index_name (str): Elasticsearch index name.
        data (list): List of dictionaries representing documents to index.
    """
    es = Elasticsearch([es_url])

    actions = [
        {"_index": index_name, "_source": doc} for doc in data
    ]
    helpers.bulk(es, actions)
    print(f"Indexed {len(data)} documents into '{index_name}'.")

if __name__ == "__main__":
    ELASTICSEARCH_URL = "http://localhost:9200"
    INDEX_NAME = "video_metadata"
    METADATA_FILE = "data/processed/video_metadata_mpg.csv"

    # Load metadata from a CSV file
    df = pd.read_csv(METADATA_FILE)

    # Convert DataFrame to a list of dictionaries
    data = df.to_dict(orient="records")

    # Index the documents
    index_documents(ELASTICSEARCH_URL, INDEX_NAME, data)
