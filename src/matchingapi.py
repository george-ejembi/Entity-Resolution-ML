from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import re
from difflib import SequenceMatcher
import jellyfish

app = FastAPI(title="Owner Reveal Entity Matching API")

# ---------------------------
# Data Models
# ---------------------------
class EntityRecord(BaseModel):
    entity_name: str
    entity_type: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

class MatchRequest(BaseModel):
    record_a: EntityRecord
    record_b: EntityRecord

class MatchResponse(BaseModel):
    similarity_score: float
    is_match: bool
    explanation: str

# ---------------------------
# Normalization Helpers
# ---------------------------
def normalize_name(name: str) -> str:
    """Standardize entity name for comparison."""
    name = name.lower().strip()
    # Remove punctuation
    name = re.sub(r'[^\w\s]', '', name)
    # Remove common legal suffixes
    suffixes = ['llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 
                'llp', 'lp', 'trust', 'holdings', 'holding', 'prop', 
                'properties', 'property', 'co', 'company']
    for suffix in suffixes:
        name = re.sub(rf'\b{suffix}\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def normalize_address(addr: str) -> str:
    """Normalize address: lower, remove punctuation, expand abbreviations."""
    if not addr:
        return ""
    addr = addr.lower().strip()
    addr = re.sub(r'[^\w\s]', '', addr)
    # Common abbreviations
    replacements = {
        'st': 'street', 'rd': 'road', 'dr': 'drive', 'ln': 'lane',
        'ct': 'court', 'crt': 'court', 'blvd': 'boulevard', 'ave': 'avenue',
        'pkwy': 'parkway', 'hwy': 'highway', 'ste': 'suite', 'apt': 'apartment'
    }
    words = addr.split()
    words = [replacements.get(w, w) for w in words]
    addr = ' '.join(words)
    addr = re.sub(r'\s+', ' ', addr).strip()
    return addr

# ---------------------------
# Similarity Functions
# ---------------------------
def name_similarity(name1: str, name2: str) -> float:
    """Combine Jaro-Winkler and token set similarity."""
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)
    if not n1 or not n2:
        return 0.0
    jw = jellyfish.jaro_winkler_similarity(n1, n2)
    # Token set similarity using Jaccard on word sets
    set1 = set(n1.split())
    set2 = set(n2.split())
    if set1 and set2:
        jaccard = len(set1 & set2) / len(set1 | set2)
    else:
        jaccard = 0.0
    return 0.7 * jw + 0.3 * jaccard

def address_similarity(addr1: str, addr2: str) -> float:
    """Normalized address comparison with Jaro-Winkler."""
    a1 = normalize_address(addr1)
    a2 = normalize_address(addr2)
    if not a1 or not a2:
        return 0.0
    return jellyfish.jaro_winkler_similarity(a1, a2)

def exact_match(val1, val2) -> float:
    """Return 1.0 if equal (case-insensitive), else 0.0."""
    if val1 is None or val2 is None:
        return 0.0
    return 1.0 if str(val1).lower().strip() == str(val2).lower().strip() else 0.0

# ---------------------------
# Composite Scoring
# ---------------------------
def compute_similarity(rec_a: EntityRecord, rec_b: EntityRecord) -> float:
    """Weighted combination of field similarities."""
    weights = {
        'name': 0.55,
        'address': 0.25,
        'city': 0.10,
        'state': 0.07,
        'type': 0.03
    }
    score = 0.0
    
    # Name
    score += weights['name'] * name_similarity(rec_a.entity_name, rec_b.entity_name)
    
    # Address
    if rec_a.address and rec_b.address:
        score += weights['address'] * address_similarity(rec_a.address, rec_b.address)
    
    # City & State (exact or near)
    if rec_a.city and rec_b.city:
        city_sim = jellyfish.jaro_winkler_similarity(rec_a.city.lower(), rec_b.city.lower())
        score += weights['city'] * city_sim
    if rec_a.state and rec_b.state:
        score += weights['state'] * exact_match(rec_a.state, rec_b.state)
    
    # Entity type (exact)
    if rec_a.entity_type and rec_b.entity_type:
        score += weights['type'] * exact_match(rec_a.entity_type, rec_b.entity_type)
    
    return min(score, 1.0)

def generate_explanation(score: float, rec_a: EntityRecord, rec_b: EntityRecord) -> str:
    """Provide human-readable reasoning for the score."""
    name_sim = name_similarity(rec_a.entity_name, rec_b.entity_name)
    addr_sim = address_similarity(rec_a.address, rec_b.address) if rec_a.address and rec_b.address else 0.0
    city_match = exact_match(rec_a.city, rec_b.city)
    state_match = exact_match(rec_a.state, rec_b.state)
    
    reasons = []
    if name_sim > 0.8:
        reasons.append("high name similarity after normalization")
    elif name_sim > 0.5:
        reasons.append("moderate name similarity")
    else:
        reasons.append("low name similarity")
    
    if addr_sim > 0.8:
        reasons.append("address is a close match")
    elif addr_sim > 0.5:
        reasons.append("address has some similarity")
    
    if city_match:
        reasons.append("same city")
    if state_match:
        reasons.append("same state")
    
    explanation = "; ".join(reasons) if reasons else "insufficient similarity"
    if score >= 0.7:
        return f"Match: {explanation}."
    else:
        return f"No match: {explanation}."

# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/match", response_model=MatchResponse)
async def match_records(request: MatchRequest):
    score = compute_similarity(request.record_a, request.record_b)
    threshold = 0.7
    is_match = score >= threshold
    explanation = generate_explanation(score, request.record_a, request.record_b)
    return MatchResponse(
        similarity_score=round(score, 4),
        is_match=is_match,
        explanation=explanation
    )

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

