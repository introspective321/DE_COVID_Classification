from elasticsearch import Elasticsearch

def create_index(es_url, index_name, mapping):
    """
    Args:
        es_url (str): Elasticsearch URL.
        index_name (str): Name of the index to be created.
        mapping (dict): Mapping configuration for the index.
    """
    es = Elasticsearch([es_url])

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

if __name__ == "__main__":
    ELASTICSEARCH_URL = "http://localhost:9200"
    INDEX_NAME = "video_metadata"

    # Define the index mapping
    MAPPING = {
        "mappings": {
            "properties": {
                "subject_id": {"type": "keyword"},
                "view": {"type": "text"},
                "file_name": {"type": "text"},
                "file_path": {"type": "text"},
                "timestamp": {"type": "date"}
            }
        }
    }

    create_index(ELASTICSEARCH_URL, INDEX_NAME, MAPPING)
