# Entity-Resolution-ML

# Owner Reveal – Entity Resolution System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Entity resolution for noisy property ownership records.**  
Identifies duplicate entities across multiple data sources and provides a real-time matching API.


# Overview

This project solves a real-world entity resolution problem: determining when multiple records refer to the same person or company despite variations in naming, formatting, and missing fields.

The system ingests records containing entity names, addresses, and types, and:

- **Deduplicates** the dataset by clustering similar records
- **Assigns confidence scores** to each cluster assignment
- **Exposes a REST API** for pairwise similarity scoring with explanations


# Approach

### 1. Blocking
To avoid comparing all record pairs (O(n²)), candidate pairs are generated using:
- Exact match on **state**
- Exact match on **city**
- Prefix of normalized **entity name** (first 5 characters)

### 2. Feature Engineering
Similarity features include:
- Fuzzy name similarity (Jaro‑Winkler, q‑gram, and token set)
- Address similarity after abbreviation expansion
- Exact matches on city, state, and entity type

### 3. Clustering
A graph‑based approach:
- Records become **nodes**
- Pairs above threshold become **edges** weighted by similarity
- **Connected components** define the final clusters

### 4. Confidence Scoring
Cluster confidence is derived from the average similarity of edges within each connected component.



## Repository Structure
```
owner-reveal-entity-resolution/
├── README.md 
├── .gitignore 
├── requirements.txt
├── data/
│ └── entity_records.csv
├── src/
│ ├── deduplicate.py 
│ └── api.py 
├── output/
│ └── clusters.csv; Generated output 
├── docs/
│ └── writeup.md # Detailed methodology & tradeoffs
└── tests/
└── test_api.py
```





