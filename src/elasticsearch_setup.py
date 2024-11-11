from elasticsearch import Elasticsearch

def setup_elasticsearch():
    """Set up Elasticsearch index configuration."""
    es = Elasticsearch()
    index_config = {
        "mappings": {
            "properties": {
                "subject_id": {"type": "keyword"},
                "temperature": {"type": "float"},
                "gender": {"type": "text"},
                "age_range": {"type": "text"},
                # Define additional fields as required...
            }
        }
    }
    es.indices.create(index="covid_health_index", body=index_config)
    print("Elasticsearch setup complete.")
