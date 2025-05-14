#!/usr/bin/env python
import requests
import time
import sys

# API endpoint
API_URL = "http://localhost:9000/forward-context"


def run_query(query, search_type="hybrid", limit=5):
    payload = {"query": query, "search_type": search_type, "limit": limit}

    print(f"Sending query: '{query}' using {search_type} search")
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            print(
                f"\nResults ({len(data['results'])} found, took {data['execution_time_ms']:.2f}ms):"
            )
            print("-" * 80)

            for i, result in enumerate(data["results"]):
                print(
                    f"{i+1}. {result['title']} (Score: {result['similarity_score']:.4f})"
                )
                print(f"   Subject: {result['subject']}")
                print(f"   Link: {result['link']}")
                print()

            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return None


# Wait for the API to be ready
def wait_for_api(max_retries=30, retry_delay=2):
    print("Waiting for API to be ready...")

    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:9000/ready")
            if response.status_code == 200:
                print("API is ready!")
                return True
            else:
                print(
                    f"API not ready yet (status: {response.status_code}), retrying..."
                )
                time.sleep(retry_delay)
        except Exception as e:
            print(f"Error connecting to API: {e}")
            time.sleep(retry_delay)

    print(f"API not ready after {max_retries} retries")
    return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = sys.argv[1]
        search_type = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
        run_query(query, search_type)
        sys.exit(0)

    if wait_for_api():
        # Test different query types
        queries = [
            {"query": "What is the gold price trend in 2024?", "search_type": "hybrid"},
            {
                "query": "What is the gold price trend in 2024?",
                "search_type": "semantic",
            },
            {
                "query": "What is the gold price trend in 2024?",
                "search_type": "keyword",
            },
        ]

        for query_data in queries:
            run_query(query_data["query"], query_data["search_type"])
            print("=" * 80)
            time.sleep(1)  # Small delay between queries
