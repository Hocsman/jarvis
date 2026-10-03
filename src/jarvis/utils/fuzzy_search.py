from __future__ import annotations
import re
from typing import List


def generate_flexible_fts_query(query: str, field_names: List[str] = None) -> str:
    """
    Generate a more flexible FTS5 query that handles variations and partial matches.
    
    Args:
        query: The search query
        field_names: Optional list of field names to search in (for multi-column FTS)
    
    Returns:
        FTS5 query string with flexible matching
    """
    if not query.strip():
        return ""
    
    # Clean and tokenize the query
    tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
    if not tokens:
        return ""
    
    # Build flexible FTS5 query components
    query_parts = []
    
    # For short queries (1-2 words), use OR logic with prefix matching
    if len(tokens) <= 2:
        prefix_terms = [f"{token}*" for token in tokens]
        exact_terms = tokens.copy()
        
        # Add both exact and prefix matches with OR
        all_terms = exact_terms + prefix_terms
        if field_names:
            # Multi-column search: search in any field
            field_parts = []
            for field in field_names:
                field_parts.extend([f"{field}:{term}" for term in all_terms])
            query_parts.append("(" + " OR ".join(field_parts) + ")")
        else:
            query_parts.append("(" + " OR ".join(all_terms) + ")")
    
    # For longer queries, use NEAR operator for phrase-like matching
    elif len(tokens) <= 5:
        # Try exact phrase first
        phrase_query = " ".join(tokens)
        
        # Add NEAR variants for flexible word order
        near_queries = []
        if len(tokens) >= 2:
            # NEAR/3 allows up to 3 words between terms
            near_queries.append(f"NEAR({' '.join(tokens)}, 3)")
        
        # Add prefix matching for the last word (common for incomplete typing)
        prefix_variant = " ".join(tokens[:-1] + [f"{tokens[-1]}*"])
        
        if field_names:
            # Multi-column search
            field_parts = []
            for field in field_names:
                field_parts.append(f'{field}:"{phrase_query}"')
                field_parts.extend([f"{field}:{nq}" for nq in near_queries])
                field_parts.append(f"{field}:{prefix_variant}")
            query_parts.append("(" + " OR ".join(field_parts) + ")")
        else:
            all_variants = [f'"{phrase_query}"'] + near_queries + [prefix_variant]
            query_parts.append("(" + " OR ".join(all_variants) + ")")
    
    # For very long queries, fall back to AND logic with some OR alternatives
    else:
        # Use first few words with AND, rest with OR
        primary_terms = tokens[:3]
        secondary_terms = tokens[3:]
        
        primary_and = " ".join(primary_terms)
        secondary_or = " OR ".join(secondary_terms)
        
        if field_names:
            field_parts = []
            for field in field_names:
                field_parts.append(f"{field}:({primary_and}) AND ({field}:({secondary_or}))")
            query_parts.append("(" + " OR ".join(field_parts) + ")")
        else:
            query_parts.append(f"({primary_and}) AND ({secondary_or})")
    
    return " OR ".join(query_parts) if query_parts else ""
