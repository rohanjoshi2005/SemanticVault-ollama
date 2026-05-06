"""
VectorDB — Python/FastAPI conversion of the C++ VectorDB project.
Implements HNSW, KD-Tree, and Brute Force search algorithms + RAG pipeline via Ollama.

Run:
    pip install fastapi uvicorn requests numpy
    python main.py
"""

import math
import time
import threading
import random
import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional
import requests
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

DIMS = 16  # demo vector dimensions

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]

DistFn = Callable[[list[float], list[float]], float]

# =====================================================================
#  DISTANCE METRICS  (numpy-accelerated)
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def cosine(a: list[float], b: list[float]) -> float:
    an, bn = np.array(a), np.array(b)
    na, nb = np.linalg.norm(an), np.linalg.norm(bn)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(an, bn) / (na * nb))

def manhattan(a: list[float], b: list[float]) -> float:
    return float(np.sum(np.abs(np.array(a) - np.array(b))))

def get_dist_fn(metric: str) -> DistFn:
    return {"cosine": cosine, "manhattan": manhattan}.get(metric, euclidean)

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")
    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None

class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        ax = depth % self.dims
        if item.emb[ax] < node.item.emb[ax]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, item: VectorItem):
        self.root = self._insert(self.root, item, 0)

    def _knn(self, node: Optional[KDNode], q: list[float], k: int, depth: int,
             dist: DistFn, heap: list):
        if node is None:
            return
        dn = dist(q, node.item.emb)
        # Max-heap with negated distance (Python's heapq is min-heap)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))

        ax = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        heap: list = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted((-d, id_) for d, id_ in heap)

    def rebuild(self, items: list[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

@dataclass
class HNSWNode:
    item: VectorItem
    max_layer: int
    neighbors: list[list[int]] = field(default_factory=list)

class HNSW:
    def __init__(self, M: int = 16, ef_build: int = 200, seed: int = 42):
        self.M = M
        self.M0 = 2 * M
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(M)
        self.graph: dict[int, HNSWNode] = {}
        self.entry_pt = -1
        self.top_layer = -1
        self._rng = random.Random(seed)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      layer: int, dist: DistFn) -> list[tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.graph[ep].item.emb)
        # candidates: min-heap by distance
        cands = [(d0, ep)]
        # found: max-heap (negated)
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst = -found[0][0]
            if len(found) >= ef and cd > worst:
                break
            node = self.graph.get(cid)
            if node is None or layer >= len(node.neighbors):
                continue
            for nid in node.neighbors[layer]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = dist(q, self.graph[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, id_) for d, id_ in found]
        result.sort()
        return result

    def insert(self, item: VectorItem, dist: DistFn):
        id_ = item.id
        lvl = self._rand_level()
        self.graph[id_] = HNSWNode(item=item, max_layer=lvl,
                                    neighbors=[[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.graph[ep].neighbors):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            selected = [id2 for _, id2 in W[:max_m]]
            self.graph[id_].neighbors[lc] = selected

            for nid in selected:
                if nid not in self.graph:
                    continue
                conn = self.graph[nid].neighbors
                while len(conn) <= lc:
                    conn.append([])
                conn[lc].append(id_)
                if len(conn[lc]) > max_m:
                    ds = sorted((dist(self.graph[nid].item.emb,
                                      self.graph[c].item.emb), c)
                                for c in conn[lc] if c in self.graph)
                    conn[lc] = [c for _, c in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: list[float], k: int, ef: int,
            dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.graph[ep].neighbors):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_: int):
        if id_ not in self.graph:
            return
        for node in self.graph.values():
            for layer in node.neighbors:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = next((nid for nid in self.graph if nid != id_), -1)
        del self.graph[id_]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes, edges = [], []
        for id_, nd in self.graph.items():
            nodes.append({"id": id_, "metadata": nd.item.metadata,
                          "category": nd.item.category, "maxLyr": nd.max_layer})
            for lc in range(min(nd.max_layer + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.neighbors):
                    for nid in nd.neighbors[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.graph),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def __len__(self):
        return len(self.graph)

# =====================================================================
#  VECTOR DATABASE  (demo 16D index — all 3 algorithms in parallel)
# =====================================================================

@dataclass
class SearchHit:
    id: int
    meta: str
    cat: str
    emb: list[float]
    dist: float

@dataclass
class SearchOut:
    hits: list[SearchHit]
    us: int           # microseconds
    algo: str
    metric: str

@dataclass
class BenchOut:
    bf_us: int
    kd_us: int
    hnsw_us: int
    n: int

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(M=16, ef_build=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(id=self._next_id, metadata=meta, category=cat, emb=emb)
            self._next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: list[float], k: int, metric: str, algo: str) -> SearchOut:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)
            hits = [SearchHit(id=id_, meta=self.store[id_].metadata,
                              cat=self.store[id_].category,
                              emb=self.store[id_].emb, dist=d)
                    for d, id_ in raw if id_ in self.store]
            return SearchOut(hits=hits, us=us, algo=algo, metric=metric)

    def benchmark(self, q: list[float], k: int, metric: str) -> BenchOut:
        with self._lock:
            dfn = get_dist_fn(metric)
            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)
            return BenchOut(
                bf_us   = timed(lambda: self.bf.knn(q, k, dfn)),
                kd_us   = timed(lambda: self.kdt.knn(q, k, dfn)),
                hnsw_us = timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                n       = len(self.store),
            )

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self.hnsw.get_info()

    def __len__(self):
        with self._lock:
            return len(self.store)

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = requests.post(f"{self.base}/api/embeddings",
                              json={"model": self.embed_model, "prompt": text},
                              timeout=30)
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(f"{self.base}/api/generate",
                              json={"model": self.gen_model, "prompt": prompt, "stream": False},
                              timeout=180)
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DOCUMENT DATABASE  (HNSW over real Ollama 768D embeddings)
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: list[float]

class DocumentDB:
    def __init__(self):
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW(M=16, ef_build=200)
        self.bf   = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self._lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self.store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self.store:
                return []
            raw = (self.bf.knn(q, k, cosine)
                   if len(self.store) < 10
                   else self.hnsw.knn(q, k, 50, cosine))
            return [(d, self.store[id_]) for d, id_ in raw
                    if id_ in self.store and d <= max_dist]

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self.store.values())

    def __len__(self):
        with self._lock:
            return len(self.store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    chunks, step = [], chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks

# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

DEMO_VECTORS = [
    # CS — dims 0-3 hot
    ("Linked List: nodes connected by pointers",              "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert",        "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal",  "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining",       "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    # Math — dims 4-7 hot
    ("Calculus: derivatives integrals and limits",            "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors",     "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    # Food — dims 8-11 hot
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls",          "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers",  "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    # Sports — dims 12-15 hot
    ("Basketball: fast-paced shooting dribbling slam dunks",   "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy",  "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game",   "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]

def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_VECTORS:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  FASTAPI APP
# =====================================================================

app = FastAPI(title="VectorDB Python")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

db     = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)
print("=== VectorDB Engine (Python) ===")
print("http://localhost:8080")
print(f"20 demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
status = "ONLINE" if ollama.is_available() else "OFFLINE"
print(f"Ollama: {status}")

# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────────

@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    try:
        q = [float(x) for x in v.split(",")]
    except ValueError:
        return JSONResponse({"error": "invalid vector"}, 400)
    if len(q) != DIMS:
        return JSONResponse({"error": f"need {DIMS}-dim vector"}, 400)

    out = db.search(q, k, metric, algo)
    return {
        "results": [{"id": h.id, "metadata": h.meta, "category": h.cat,
                     "embedding": h.emb, "distance": round(h.dist, 4)}
                    for h in out.hits],
        "algo": out.algo, "metric": out.metric, "timeUs": out.us,
    }

@app.post("/insert")
async def insert(req: Request):
    body = await req.json()
    meta = body.get("metadata", "")
    cat  = body.get("category", "")
    emb  = body.get("embedding", [])
    if not meta or not emb:
        return JSONResponse({"error": "need metadata and embedding"}, 400)
    dist = get_dist_fn(body.get("metric", "cosine"))
    id_  = db.insert(meta, cat, emb, dist)
    return {"id": id_}

@app.delete("/delete/{id_}")
def delete(id_: int):
    ok = db.remove(id_)
    return {"ok": ok}

@app.get("/items")
def items():
    all_items = db.all()
    return [{"id": v.id, "metadata": v.metadata, "category": v.category,
             "embedding": v.emb} for v in all_items]

@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    try:
        q = [float(x) for x in v.split(",")]
    except ValueError:
        return JSONResponse({"error": "invalid vector"}, 400)
    b = db.benchmark(q, k, metric)
    return {"bruteforceUs": b.bf_us, "kdtreeUs": b.kd_us,
            "hnswUs": b.hnsw_us, "n": b.n}

@app.get("/hnsw-info")
def hnsw_info():
    return db.hnsw_info()

@app.get("/stats")
def stats():
    return {"count": len(db), "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"]}

# ── DOCUMENT + RAG ENDPOINTS ──────────────────────────────────────────

@app.post("/doc/insert")
async def doc_insert(req: Request):
    body   = await req.json()
    title  = body.get("title", "")
    text   = body.get("text", "")
    if not title or not text:
        return JSONResponse({"error": "need title and text"}, 400)

    chunks = chunk_text(text, 250, 30)
    ids    = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return JSONResponse({"error": (
                "Ollama unavailable. Install from https://ollama.com then run: "
                "ollama pull nomic-embed-text && ollama pull llama3.2")}, 503)
        chunk_title = (f"{title} [{i+1}/{len(chunks)}]"
                       if len(chunks) > 1 else title)
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return {"ids": ids, "chunks": len(chunks), "dims": doc_db.dims}

@app.get("/doc/list")
def doc_list():
    docs = doc_db.all()
    return [{"id": d.id, "title": d.title,
             "preview": d.text[:120] + ("…" if len(d.text) > 120 else ""),
             "words": len(d.text.split())} for d in docs]

@app.delete("/doc/delete/{id_}")
def doc_delete(id_: int):
    ok = doc_db.remove(id_)
    return {"ok": ok}

@app.post("/doc/search")
async def doc_search(req: Request):
    body     = await req.json()
    question = body.get("question", "")
    k        = body.get("k", 3)
    if not question:
        return JSONResponse({"error": "need question"}, 400)
    q_emb = ollama.embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"}, 503)
    hits = doc_db.search(q_emb, k)
    return {"contexts": [{"id": doc.id, "title": doc.title,
                          "distance": round(d, 4)} for d, doc in hits]}

@app.post("/doc/ask")
async def doc_ask(req: Request):
    body     = await req.json()
    question = body.get("question", "")
    k        = body.get("k", 3)
    if not question:
        return JSONResponse({"error": "need question"}, 400)

    q_emb = ollama.embed(question)
    if not q_emb:
        return JSONResponse({"error": "Ollama unavailable"}, 503)

    hits = doc_db.search(q_emb, k)

    ctx = "\n\n".join(f"[{i+1}] {doc.title}:\n{doc.text}"
                      for i, (_, doc) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )
    answer = ollama.generate(prompt)

    return {
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [{"id": doc.id, "title": doc.title, "text": doc.text,
                      "distance": round(d, 4)} for d, doc in hits],
        "docCount": len(doc_db),
    }

@app.get("/status")
def status():
    up = ollama.is_available()
    return {"ollamaAvailable": up,
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": len(doc_db),
            "docDims": doc_db.dims,
            "demoDims": DIMS,
            "demoCount": len(db)}

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>index.html not found</h1>", 404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
