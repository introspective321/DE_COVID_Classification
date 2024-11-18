from elasticsearch import Elasticsearch

def search_documents(es_url, index_name, query):
    """
    Search for documents in Elasticsearch based on a query.

    Args:
        es_url (str): Elasticsearch URL.
        index_name (str): Elasticsearch index name.
        query (dict): Query DSL for Elasticsearch.

    Returns:
        list: List of search results.
    """
    es = Elasticsearch([es_url])
    response = es.search(index=index_name, body=query)
    return response["hits"]["hits"]

if __name__ == "__main__":
    ELASTICSEARCH_URL = "http://localhost:9200"
    INDEX_NAME = "video_metadata"

    # Define a sample search query
    QUERY = {
        "query": {
            "match": {
                "view": "Front"
            }
        }
    }

    # Perform the search
    results = search_documents(ELASTICSEARCH_URL, INDEX_NAME, QUERY)
    for result in results:
        print(result["_source"])
