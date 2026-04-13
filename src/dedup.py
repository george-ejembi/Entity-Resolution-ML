# Importing of libraries
import pandas as pd
import numpy as np
import recordlinkage as rl
from recordlinkage.preprocessing import clean, phonenumbers
import re
import networkx as nx
from typing import Dict, Any

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    
    # Standardize column names 
    df.columns = ['record_id', 'entity_name', 'entity_type', 'address', 
                  'city', 'state', 'source']
    
    # Clean and normalize text fields
    for col in ['entity_name', 'address', 'city']:
        df[col] = df[col].astype(str).fillna('')
        df[col] = df[col].str.lower().str.strip()
        df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))
    
    # Normalize state to uppercase two-letter codes
    df['state'] = df['state'].astype(str).str.upper().str.strip()
    
    # Creating a normalized name column by removing common suffixes
    df['name_clean'] = df['entity_name'].apply(normalize_entity_name)
    
    return df

def normalize_entity_name(name: str) -> str:
    """Remove common legal entity suffixes and punctuation."""
    suffixes = ['llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 
                'llp', 'lp', 'trust', 'holdings', 'holding', 'prop', 
                'properties', 'property', 'co', 'company']
    for suffix in suffixes:
        name = re.sub(rf'\b{suffix}\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[.,]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

# ---------------------------
# Blocking
# ---------------------------
def create_blocking_index(df: pd.DataFrame) -> pd.MultiIndex:
    """Create candidate pairs using multiple blocking keys."""
    indexer = rl.Index()
    
    # Block on city + state (exact match)
    indexer.block(['city', 'state'])
    
    # Additional block on first 5 chars of cleaned name
    df['name_prefix'] = df['name_clean'].str[:5]
    indexer.block('name_prefix')
    
    candidate_pairs = indexer.index(df)
    return candidate_pairs

# ---------------------------
# Feature Comparison
# ---------------------------
def compute_comparison_vectors(df: pd.DataFrame, pairs: pd.MultiIndex) -> pd.DataFrame:
    """Compute similarity features for candidate pairs."""
    compare = rl.Compare()
    
    # Name similarities
    compare.string('entity_name', 'entity_name', method='jarowinkler', label='name_jw')
    compare.string('name_clean', 'name_clean', method='jarowinkler', label='name_clean_jw')
    compare.string('entity_name', 'entity_name', method='qgram', label='name_qgram')  # q=3 is default
    
    # Address similarities
    compare.string('address', 'address', method='jarowinkler', label='addr_jw')
    compare.string('address', 'address', method='qgram', label='addr_qgram')          # q=3 is default
    
    # Exact matches on categorical fields
    compare.exact('entity_type', 'entity_type', label='type_exact')
    compare.exact('city', 'city', label='city_exact')
    compare.exact('state', 'state', label='state_exact')
    
    features = compare.compute(pairs, df)
    return features

# ---------------------------
# Scoring & Classification
# ---------------------------
def score_pairs(features: pd.DataFrame) -> pd.Series:
    """Convert features to a similarity score (0-1) using weighted rules."""
    weights = {
        'name_clean_jw': 0.35,
        'name_jw': 0.15,
        'name_qgram': 0.10,
        'addr_jw': 0.20,
        'addr_qgram': 0.10,
        'type_exact': 0.05,
        'city_exact': 0.03,
        'state_exact': 0.02
    }
    score = pd.Series(0.0, index=features.index)
    for col, weight in weights.items():
        if col in features.columns:
            score += features[col].fillna(0) * weight
    return score.clip(0, 1)

def get_confidence(score: float, threshold: float = 0.7) -> float:
    """Map score to confidence, with higher confidence farther from threshold."""
    if score >= threshold:
        # For matches, confidence increases as score approaches 1
        return 0.5 + 0.5 * (score - threshold) / (1 - threshold)
    else:
        # For non-matches, confidence would be lower, but we only output matches
        return score # will not be used for clusters.csv

# ---------------------------
# Clustering
# ---------------------------
def create_clusters(df: pd.DataFrame, pair_scores: pd.Series, threshold: float = 0.7) -> pd.DataFrame:
    """Build connected components from pairs above threshold."""
    G = nx.Graph()
    G.add_nodes_from(df.index)
    
    # Add edges for pairs above threshold
    for (idx1, idx2), score in pair_scores.items():
        if score >= threshold:
            G.add_edge(idx1, idx2, weight=score)
    
    # Find connected components (clusters)
    clusters = {}
    for cluster_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            clusters[node] = cluster_id
    
    # Assigning confidence for each record, take max score among its edges in cluster
    confidence = {}
    for node in df.index:
        if node in clusters:
            cluster_nodes = [n for n in G.neighbors(node) if G.nodes[n].get('cluster') == clusters[node]]
            if cluster_nodes:
                max_score = max(pair_scores.get((node, neighbor), 0) for neighbor in cluster_nodes)
                confidence[node] = get_confidence(max_score, threshold)
            else:
                confidence[node] = 0.5 
        else:
            confidence[node] = 0.0 
    
    # Build output DataFrame
    result = pd.DataFrame({
        'record_id': df['record_id'],
        'cluster_id': [clusters.get(i, -1) for i in df.index],
        'confidence_score': [confidence.get(i, 0.0) for i in df.index]
    })
    # Assign singleton clusters a unique ID (-1 means no cluster)
    max_cluster = max(clusters.values()) if clusters else -1
    for i, row in result.iterrows():
        if row['cluster_id'] == -1:
            max_cluster += 1
            result.at[i, 'cluster_id'] = max_cluster
    return result

# ---------------------------
# Main Execution
# ---------------------------
def main():
    print("Loading and cleaning data")
    df = load_and_clean_data (r"C:\Users\USER\Downloads\entity_records.csv")
    print(f"Loaded {len(df)} records.")
    
    print("Generating candidate pairs via blocking")
    candidate_pairs = create_blocking_index(df)
    print(f"Generated {len(candidate_pairs)} candidate pairs.")
    
    print("Computing comparison features")
    features = compute_comparison_vectors(df, candidate_pairs)
    
    print("Scoring pairs")
    pair_scores = score_pairs(features)
    
    print("Clustering records")
    clusters_df = create_clusters(df, pair_scores, threshold=0.7)
    
    print("Saved clusters.csv")
    clusters_df.to_csv('clusters.csv', index=False)
    print("Done. Output written to clusters.csv")

if __name__ == '__main__':
    main()
