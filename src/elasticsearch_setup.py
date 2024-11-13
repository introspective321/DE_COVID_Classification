from elasticsearch import Elasticsearch

def setup_elasticsearch(es_url):
    """Create an Elasticsearch index."""
    es = Elasticsearch(hosts=[es_url])
    index_config = {
        "mappings": {
            "properties": {
                "subject_id": {"type": "keyword"},
                "gender": {"type": "text"},
                "age": {"type": "integer"},
                "date": {"type": "date"},
                "t_cr_max": {"type": "float"},
                "t_cl_max": {"type": "float"}
            }
        }
    }
    es.indices.create(index="covid_health_index", body=index_config, ignore=400)
    print("Elasticsearch index setup complete.")
