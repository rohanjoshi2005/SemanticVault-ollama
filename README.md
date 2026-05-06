# VectorDB — Python/FastAPI Edition

A Python conversion of the original C++ VectorDB project.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms + a **RAG pipeline** via Ollama.  
The original `index.html` frontend works unchanged — the REST API is identical.

---

## What Changed vs the C++ Version

| Aspect | C++ | Python |
|---|---|---|
| HTTP server | `httplib.h` (single-header) | FastAPI + uvicorn |
| JSON parsing | Manual string parsing (~100 lines) | Built-in (automatic) |
| Ollama client | Raw HTTP with httplib | `requests` library |
| Distance math | Manual loops | `numpy` (faster for large vectors) |
| Threading | `std::mutex` | `threading.Lock()` |
| Build step | `g++ -std=c++17 -O2 main.cpp -o db -lws2_32` | None — just `python main.py` |
| Binary | `db.exe` | N/A |

**Algorithm logic is identical** — HNSW, KD-Tree, and BruteForce are direct ports
of the C++ code with the same parameters (M=16, ef_build=200, etc.).

---

## Prerequisites

1. **Python 3.11+**
2. **Ollama** — https://ollama.com (for real embeddings and RAG)

---

## Setup

### Step 1 — Install Python dependencies

```bash
pip install fastapi uvicorn requests numpy
```

### Step 2 — Install Ollama models (for the Documents + RAG tabs)

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Step 3 — Run the server

```bash
python main.py
```

You should see:
```
=== VectorDB Engine (Python) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
```

### Step 4 — Open the browser

```
http://localhost:8080
```

The frontend is the same `index.html` as the original project.

---

## Project Structure

```
VectorDB/
├── main.py       ← Python backend (HNSW, KD-Tree, BruteForce, FastAPI, RAG)
├── index.html    ← Frontend — unchanged from original C++ project
└── README.md     ← This file
```

## REST API

Identical to the C++ version — all endpoints are preserved:

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/{id}` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `GET` | `/stats` | DB statistics |

### Document & RAG Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/doc/insert` | Embed and store document |
| `GET` | `/doc/list` | List all stored document chunks |
| `DELETE` | `/doc/delete/{id}` | Delete document chunk |
| `POST` | `/doc/ask` | RAG: retrieve + generate answer |
| `GET` | `/status` | Ollama status and model info |

---

## Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install fastapi uvicorn requests numpy` |
| `Ollama: OFFLINE` | Run `ollama serve` in a terminal |
| Port 8080 in use | Edit the last line of `main.py`: `uvicorn.run(app, port=9090)` |
| Slow LLM responses | Switch to `llama3.2:1b` — edit `ollama.gen_model` in `main.py` |

---

## Performance Note

Python is ~5–10x slower than C++ for the algorithm internals (HNSW graph traversal, KD-Tree recursion).
At 20 demo vectors this is completely imperceptible.
At 10,000+ vectors with 768D embeddings, the C++ version would be noticeably faster.
For production scale, use [hnswlib](https://github.com/nmslib/hnswlib) (C++ bindings for Python).
