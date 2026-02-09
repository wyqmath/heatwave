#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np
import requests

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: str = "config/default.yaml") -> dict:
    cfg_path = Path(path)
    if yaml is None or not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_nodes(csv_path: Path) -> list:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        idx = header.index("Node") if "Node" in header else 0
        nodes = [row[idx].strip() for row in reader if len(row) > idx and row[idx].strip()]
    return nodes


class NodeRecommender:
    def __init__(self, csv_path="all_nodes.csv", embed_path="embeddings.npy", index_path=None, config=None):
        self.config = config or load_config()
        node_cfg = self.config.get("node_recommender", {})
        qa_cfg = self.config.get("qa_engine", {})
        self.csv_path = Path(csv_path or qa_cfg.get("nodes_csv_path", "all_nodes.csv"))
        self.embed_path = Path(embed_path or node_cfg.get("embeddings_file", "embeddings.npy"))
        self.index_path = Path(index_path or node_cfg.get("faiss_index_file", f"{self.embed_path}.faiss"))
        self.model = node_cfg.get("embedding_model", "text-embedding-v4")
        self.base_url = node_cfg.get("embedding_api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.api_key = node_cfg.get("embedding_api_key") or os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.dimension = int(node_cfg.get("embedding_dimension", 1024))
        self.batch_size = int(node_cfg.get("embedding_batch_size", self.config.get("compute", {}).get("embedding_batch_size", 10)))
        rpm = int(node_cfg.get("requests_per_minute", self.config.get("llm", {}).get("requests_per_minute", 1800)))
        self.request_interval = 60.0 / max(rpm, 1)
        self.last_request_time = 0.0
        self.nodes = read_nodes(self.csv_path)
        self.embeddings = self._load_or_build_embeddings()
        self.index = self._load_or_build_index()

    def _throttle(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self.last_request_time = time.time()

    def _request_embeddings(self, texts, max_retries=5):
        if not self.api_key:
            raise ValueError("Missing embedding API key.")
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "input": texts, "encoding_format": "float"}

        for attempt in range(1, max_retries + 1):
            if attempt == 1 and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("API Config - URL: %s, Model: %s, Batch size: %d", url, self.model, len(texts))
                logging.debug("API Key (first 10 chars): %s...", self.api_key[:10] if self.api_key else "None")
            try:
                self._throttle()
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                if len(data) != len(texts):
                    raise ValueError("Embedding count mismatch.")
                return [item["embedding"] for item in data]
            except requests.HTTPError as exc:
                wait_s = min(2 ** attempt, 10)
                try:
                    error_detail = exc.response.json()
                    logging.warning("Embedding API Error (attempt %d/%d): %s", attempt, max_retries, error_detail)
                    if attempt == 1:
                        logging.warning("Request details - Model: %s, Texts count: %d, First text: %s",
                                      self.model, len(texts), texts[0][:50] if texts else "N/A")
                except:
                    logging.warning("Embedding request failed (attempt %d): %s - Response: %s",
                                  attempt, exc, exc.response.text if hasattr(exc, 'response') else 'N/A')
                time.sleep(wait_s)
            except Exception as exc:
                wait_s = min(2 ** attempt, 10)
                logging.warning("Embedding request failed (attempt %d): %s", attempt, exc)
                time.sleep(wait_s)

        logging.error("All %d attempts failed, returning zero vectors", max_retries)
        return [np.zeros(self.dimension).tolist() for _ in texts]

    def _build_embeddings(self) -> np.ndarray:
        embeddings = []
        total = len(self.nodes)
        for i in range(0, total, self.batch_size):
            if i % (self.batch_size * 10) == 0:
                logging.info("Embedding %d/%d", i, total)
            batch = self.nodes[i : i + self.batch_size]
            embeddings.extend(self._request_embeddings(batch))
        arr = np.asarray(embeddings, dtype="float32")
        np.save(self.embed_path, arr)
        return arr

    def _load_or_build_embeddings(self) -> np.ndarray:
        if self.embed_path.exists():
            arr = np.load(self.embed_path)
            if arr.shape[0] == len(self.nodes):
                return arr.astype("float32")
            logging.warning("Embedding size mismatch, rebuilding.")
        return self._build_embeddings()

    def _load_or_build_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        vectors = self.embeddings.astype("float32")
        if vectors.shape[1] != self.dimension:
            self.dimension = vectors.shape[1]
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(self.dimension)
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))
        return index

    def get_top_nodes(self, query: str, top_k: int = 5):
        q_vec = np.asarray(self._request_embeddings([query])[0], dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q_vec)
        distances, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.nodes):
                results.append({"Node": self.nodes[idx], "similarity": float(score)})
        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Build node embeddings and FAISS index.")
    parser.add_argument("--csv", default=None, help="Path to all_nodes.csv")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.npy")
    parser.add_argument("--index", default=None, help="Path to FAISS index")
    parser.add_argument("--query", default=None, help="Query text to test retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    recommender = NodeRecommender(csv_path=args.csv, embed_path=args.embeddings, index_path=args.index)

    if args.query:
        for i, item in enumerate(recommender.get_top_nodes(args.query, args.top_k), 1):
            logging.info("%d. %s (%.4f)", i, item["Node"], item["similarity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())