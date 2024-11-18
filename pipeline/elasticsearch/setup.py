from elasticsearch import Elasticsearch

def create_index(es_url, index_name, mapping):
    es = Elasticsearch([es_url])
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists.")
    else:
        es.indices.create(index=index_name, body=mapping)
        print(f"Index '{index_name}' created successfully.")

if __name__ == "__main__":
    ELASTICSEARCH_URL = "http://localhost:9200"
    INDEX_NAME = "video_metadata"
    MAPPING = {
        "mappings": {
            "properties": {
                "subject_id": {"type": "keyword"},
                "view": {"type": "text"},
                "file_name": {"type": "text"},
                "file_path": {"type": "text"},
                "duration": {"type": "float"},
                "thermal_quality": {"type": "text"}
            }
        }
    }
    create_index(ELASTICSEARCH_URL, INDEX_NAME, MAPPING)
