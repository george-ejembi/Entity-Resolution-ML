import requests

url = "http://127.0.0.1:8000/match"
payload = {
    "record_a": {
        "entity_name": "Lone Star Property Holdings LLC",
        "entity_type": "company",
        "address": "100 Crescent Ct",
        "city": "Dallas",
        "state": "TX"
    },
    "record_b": {
        "entity_name": "LONE STAR PROP HOLDINGS",
        "entity_type": "company",
        "address": "100 Crescent Court",
        "city": "Dallas",
        "state": "TX"
    }
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
