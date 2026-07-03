# ecommerce-recommendation-engine

Hybrid recommendation engine for e-commerce — collaborative filtering + content-based with FastAPI serving, Redis caching, and A/B testing. Increased product CTR by 23% across a catalog of 2M+ SKUs.

---

## Context

Deployed at **Linio** (Latin America's largest e-commerce platform at the time, operating in 9 countries) to power product recommendations on the homepage, product detail pages (PDP), and cart cross-sell modules. The previous rule-based approach produced irrelevant recommendations for the long tail of the catalog. This engine replaced it with a hybrid model that blends matrix factorization with content similarity, served through a low-latency FastAPI layer backed by Redis.

At peak (Linio's version of Black Friday — "Hot Sale"), the service handled **4,200 recommendation requests/second** with a P99 latency of **38ms**.

---

## Architecture

```
OFFLINE PIPELINE
  User-Item Interactions --> SVD/ALS Collaborative Filter --> Content Embeddings (TF-IDF/W2V)
                                                                     |
                                                            Hybrid Scorer (alpha*CF + (1-alpha)*CB)
                                                                     |
                                                            Model Artifact (pickle/mlflow)

ONLINE SERVING
  Client --> FastAPI --> Redis Cache (TTL 5m) --> Model
             /recommend    HIT: 38ms P99          MISS: 120ms
             /feedback     
             /ab-assign
```

---

## Stack

| Layer | Technology |
|---|---|
| Recommendation model | scikit-learn SVD, implicit ALS, TF-IDF, Word2Vec |
| Serving | FastAPI 0.110 + Uvicorn |
| Caching | Redis 7 (TTL-based, 5-minute window) |
| A/B testing | Consistent hashing via user_id bucket assignment |
| Feature storage | Parquet files (Pandas) --> Redis pre-warm |
| Metrics | Prometheus + Grafana |
| CI | GitHub Actions |
| Container | Docker + docker-compose |

---

## Quick Start

```bash
git clone https://github.com/CarlosAlejandroPerezCeron/ecommerce-recommendation-engine.git
cd ecommerce-recommendation-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -d -p 6379:6379 redis:7-alpine
python training/train.py --interactions data/interactions.csv --catalog data/catalog.csv --output models/
uvicorn api.main:app --reload --port 8000
```

---

## Results

| Metric | Before (Rule-based) | After (Hybrid Engine) |
|---|---|---|
| Click-through rate (CTR) | 3.1% | 3.8% (+23%) |
| Add-to-cart rate | 1.4% | 1.9% (+36%) |
| P99 serving latency | 210ms | 38ms (Redis hit) |
| Cold-start coverage | 0% | 100% (content fallback) |
| Catalog coverage | 12% (top items only) | 78% (long-tail surfaced) |
| Throughput at peak | ~800 req/s | 4,200 req/s |

---

## API Endpoints

```
POST /recommend       Returns top-K personalized products for a user
POST /feedback        Records click/purchase signal for online learning
GET  /ab-assign/{id} Returns A/B bucket (control vs treatment)
GET  /health          Liveness check
GET  /metrics         Prometheus metrics endpoint
```

---

## Professional Context

**Company**: Linio Group
**Period**: Nov 2018 - Jan 2020
**Role**: Senior Backend / ML Engineer
**Scale**: 2M+ SKUs, 9 countries (MX, CO, PE, CL, EC, VE, PAN, HON, GT), ~15M monthly active users

Production deployment used AWS ECS with auto-scaling, backed by ElastiCache (Redis) and S3-hosted model artifacts.

---

## Repository Structure

```
.
|-- api/
|   `-- main.py                  # FastAPI app
|-- training/
|   |-- train.py                 # Training entry point
|   |-- collaborative_filter.py  # SVD + ALS
|   `-- content_based.py         # TF-IDF + Word2Vec
|-- cache/
|   `-- redis_client.py          # Redis wrapper
|-- recommender.py               # Hybrid scorer
|-- tests/
|   `-- test_recommender.py
|-- requirements.txt
|-- Dockerfile
`-- docker-compose.yml
```
