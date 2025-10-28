"""
processors/retrieval.py
----------------------------------------
Load artwork metadata + embeddings,
compare a user pose embedding against them,
and return the Top-K matches.
"""

import os
import torch
import pandas as pd
import numpy as np
from .similarity import compute_cosine_scores, top_k_indices


class ArtworkDatabase:
    def __init__(self, csv_path="data/portrait_works.csv"):
        """
        Loads the metadata table (portrait_works.csv) into memory.
        Embeddings themselves are loaded lazily when needed.
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def load_all_embeddings(self):
        """
        Loads all artwork embeddings from the 'embedding_path' column.
        Returns:
            embeddings_matrix: np.ndarray of shape (N, D)
            rows_list: list[dict-like] with row info for each embedding
        We skip rows where the embedding file is missing.
        """
        embs = []
        rows = []

        for _, row in self.df.iterrows():
            emb_path = row["embedding_path"]
            if not os.path.exists(emb_path):
                continue
            emb = torch.load(emb_path).cpu().numpy()
            # emb is typically shape (1, D), we flatten to (D,)
            emb = emb.reshape(-1)
            embs.append(emb)
            rows.append(row)

        if len(embs) == 0:
            raise RuntimeError("No artwork embeddings found. Did you run generate_embeddings.py?")

        embeddings_matrix = np.vstack(embs)
        return embeddings_matrix, rows

    def retrieve_top_k(self, user_embedding, k=3):
        """
        Given a user pose embedding (np.ndarray shape (1,D) or (D,)),
        compute cosine similarity to all artwork embeddings,
        and return the top-k matches with metadata.

        Returns:
            results: list of dict with keys:
                'rank', 'artist', 'title', 'score', 'notes_pose', 'row'
        """
        all_embs, rows = self.load_all_embeddings()

        scores = compute_cosine_scores(user_embedding, all_embs)
        best_idxs = top_k_indices(scores, k=k)

        results = []
        for rank, idx in enumerate(best_idxs, start=1):
            row = rows[idx]
            result = {
                "rank": rank,
                "artist": row["artist_en"],
                "title": row["title_en"],
                "year": row["year"],
                "score": float(scores[idx]),
                "notes_pose": row["notes_pose"],
                "file_name": row["file_name"],
                "row": row,  # full metadata row if caller needs more
            }
            results.append(result)

        return results
