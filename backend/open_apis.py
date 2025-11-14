# backend/open_apis.py
"""
Global open-collection search helpers.

This module shows how EAR could talk to real museum APIs
(Tier 1 / Tier 2 as described in the README).

For the live demo we keep it simple:
- Implement a real call to The Met Museum API as an example.
- Other sources can be added with the same unified interface.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import requests


@dataclass
class GlobalArtwork:
    source: str          # e.g. "met"
    source_id: str       # e.g. Met objectID
    title: str
    artist: str
    year: str
    image_url: str
    museum: str
    country: str = ""
    permalink: str = ""  # museum's page URL


# ------------- The Met Museum (example Tier 1 API) -------------


def _met_search_ids(query: str, max_results: int = 20) -> List[int]:
    """
    Call The Met Museum search API.
    Docs: https://metmuseum.github.io/
    """
    url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    params = {
        "q": query,
        "hasImages": "true",
        "isOnView": "true",
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    ids = data.get("objectIDs") or []
    if not isinstance(ids, list):
        return []
    return ids[:max_results]


def _met_fetch_object(obj_id: int) -> Optional[GlobalArtwork]:
    url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()

    image = data.get("primaryImage") or data.get("primaryImageSmall") or ""
    if not image:
        return None  # we only care about records with images

    title = data.get("title") or ""
    artist = data.get("artistDisplayName") or ""
    year = data.get("objectDate") or ""
    museum = "The Metropolitan Museum of Art"
    country = data.get("country") or ""
    permalink = data.get("objectURL") or ""

    return GlobalArtwork(
        source="met",
        source_id=str(obj_id),
        title=title,
        artist=artist,
        year=year,
        image_url=image,
        museum=museum,
        country=country,
        permalink=permalink,
    )


def search_met(query: str, topk: int = 10) -> List[Dict[str, Any]]:
    """
    High-level wrapper: from text query to normalized artworks.
    """
    results: List[Dict[str, Any]] = []
    try:
        ids = _met_search_ids(query, max_results=topk)
    except Exception:
        return results

    for oid in ids:
        try:
            art = _met_fetch_object(int(oid))
        except Exception:
            art = None
        if art:
            results.append(asdict(art))
    return results


# ------------- Aggregator entry point -------------


def search_global_apis(query: str, topk_per_source: int = 10) -> Dict[str, Any]:
    """
    Unified interface used by FastAPI endpoint.

    For the prototype we only call The Met as a concrete example.
    In the README we describe how to expand this to:
    - Art Institute of Chicago
    - Cleveland Museum of Art
    - Harvard Art Museums
    - DPLA / Europeana aggregators
    etc.
    """
    all_results: Dict[str, Any] = {}

    # Tier 1: The Met Museum
    met_items = search_met(query, topk_per_source)
    all_results["met"] = met_items

    # TODO (future work): add more sources, e.g.
    # all_results["aic"] = search_aic(query, topk_per_source)
    # all_results["dpla"] = search_dpla(query, topk_per_source)
    # all_results["europeana"] = search_europeana(query, topk_per_source)

    return {
        "query": query,
        "sources": all_results,
    }
