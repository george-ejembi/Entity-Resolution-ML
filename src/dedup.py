# ---------------------------
# IMPORTS
# ---------------------------
import pandas as pd
import recordlinkage as rl
import re
import networkx as nx

# ---------------------------
# DATA LOADING & CLEANING
# ---------------------------
def normalize_entity_name(name: str) -> str:
    suffixes = [
        'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited',
        'llp', 'lp', 'trust', 'holdings', 'holding',
        'prop', 'properties', 'property', 'co', 'company'
    ]
    for suffix in suffixes:
        name = re.sub(rf'\b{suffix}\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[.,]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df.columns = [
        'record_id', 'entity_name', 'entity_type',
        'address', 'city', 'state', 'source'
    ]

    for col in ['entity_name', 'address', 'city']:
        df[col] = df[col].astype(str).fillna('')
        df[col] = df[col].str.lower().str.strip()
        df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x))

    df['state'] = df['state'].astype(str).str.upper().str.strip()

    df['name_clean'] = df['entity_name'].apply(normalize_entity_name)

    return df


# ---------------------------
# BLOCKING
# ---------------------------
def create_blocking_index(df: pd.DataFrame) -> pd.MultiIndex:
    indexer = rl.Index()

    indexer.block(['city', 'state'])

    df['name_prefix'] = df['name_clean'].str[:5]
    indexer.block('name_prefix')

    return indexer.index(df)


# ---------------------------
# FEATURE COMPARISON
# ---------------------------
def compute_comparison_vectors(df, pairs):
    compare = rl.Compare()

    compare.string('entity_name', 'entity_name', method='jarowinkler', label='name_jw')
    compare.string('name_clean', 'name_clean', method='jarowinkler', label='name_clean_jw')
    compare.string('entity_name', 'entity_name', method='qgram', label='name_qgram')

    compare.string('address', 'address', method='jarowinkler', label='addr_jw')
    compare.string('address', 'address', method='qgram', label='addr_qgram')

    compare.exact('entity_type', 'entity_type', label='type_exact')
    compare.exact('city', 'city', label='city_exact')
    compare.exact('state', 'state', label='state_exact')

    return compare.compute(pairs, df)


# ---------------------------
# SCORING
# ---------------------------
def score_pairs(features):
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


# ---------------------------
# CLUSTERING
# ---------------------------
def create_clusters(df, pair_scores, threshold=0.65):

    G = nx.Graph()
    G.add_nodes_from(df.index)

    # Add edges
    for (i, j), score in pair_scores.items():
        if score >= threshold:
            G.add_edge(i, j, weight=score)

    # Connected components
    clusters = {}
    for cid, component in enumerate(nx.connected_components(G)):
        for node in component:
            clusters[node] = cid

    # ---------------------------
    # FIXED CONFIDENCE LOGIC
    # ---------------------------
    confidence = {}

    for node in df.index:
        neighbors = list(G.neighbors(node))

        if neighbors:
            scores = []

            for neighbor in neighbors:
                s = pair_scores.get((node, neighbor)) or pair_scores.get((neighbor, node))
                if s is not None:
                    scores.append(s)

            if scores:
                confidence[node] = round(sum(scores) / len(scores), 2)
            else:
                confidence[node] = 0.0
        else:
            confidence[node] = 0.0

    # ---------------------------
    # BUILD OUTPUT
    # ---------------------------
    result = pd.DataFrame({
        'record_id': df['record_id'],
        'cluster_id': [clusters.get(i, -1) for i in df.index],
        'confidence_score': [confidence.get(i, 0.0) for i in df.index]
    })

    # Handle singletons
    max_cluster = max(clusters.values()) if clusters else -1

    for i in result.index:
        if result.loc[i, 'cluster_id'] == -1:
            max_cluster += 1
            result.loc[i, 'cluster_id'] = max_cluster

    return result


# ---------------------------
# MAIN
# ---------------------------
def main():
    print("Loading data")
    df = load_and_clean_data(r"C:/Users/USER/Downloads/entity_records.csv")

    print(f"{len(df)} records loaded")

    print("Blocking")
    pairs = create_blocking_index(df)
    print(f"{len(pairs)} candidate pairs")

    print("Computing features")
    features = compute_comparison_vectors(df, pairs)

    print("Scoring")
    scores = score_pairs(features)

    print(scores.describe()) 
    print("Clustering")
    clusters_df = create_clusters(df, scores, threshold=0.65)

    clusters_df.to_csv("clusters.csv", index=False)

    print("Done. clusters.csv generated.")


if __name__ == "__main__":
    main()
