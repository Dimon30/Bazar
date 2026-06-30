# Bazar  
**End-to-End ML System for Marketplace Ranking & Dynamic Pricing**

---

## Overview

Bazar is a production-style ML project that simulates a decision system for an online marketplace.

The system models the full pipeline — from a user search query to final ranking and pricing decisions — demonstrating how machine learning can be applied to improve relevance, conversion, and revenue in e-commerce.

The project focuses on:

- Learning-to-Rank for search relevance  
- Dynamic pricing based on historical demand signals  
- Reproducible ML pipelines  
- Offline evaluation aligned with business metrics  
- Production-oriented engineering practices  

---

## Business Problem

Modern online marketplaces often lose revenue due to two key issues:

1. **Irrelevant search results**, leading to lower click-through rates and poor user experience.  
2. **Static pricing strategies**, which fail to adapt to demand, seasonality, and market conditions.  

Bazar addresses these problems by:

- Ranking products by relevance to user search queries using learning-to-rank models optimized with NDCG@K.  
- Dynamically adjusting product prices using quantile-based regression and business-constrained decision logic.  

The system simulates a realistic ML-driven marketplace workflow, bridging the gap between modeling and production-style deployment.

---

## System Architecture

### 1. Search Ranking Module

- Group-aware train/validation split (by `query_id`)  
- Feature engineering (text + statistical features)  
- CatBoostRanker (YetiRank loss)  
- Offline evaluation using NDCG@10  
- Model artifact versioning  

### 2. Dynamic Pricing Module

- Lag-based and rolling statistical features  
- Quantile regression for price band prediction  
- Mid-price and log-width modeling  
- Decision-layer with business constraints  
- Reproducible training pipeline  

---

## ML Engineering Principles

The project follows production-oriented ML practices:

- Explicit train/validation split to prevent data leakage  
- Offline evaluation aligned with ranking business metrics  
- Configuration-driven training  
- Artifact saving (model + metadata)  
- Modular project structure  
- Prepared inference interface (API-ready design)  

---

## Scope & Limitations

This project is a simulation of a real-world marketplace ML system and is designed primarily for educational and demonstrational purposes.

Key limitations include:

- Models are trained on public or synthetic datasets and may not fully reflect real production data distributions.  
- Offline metrics such as NDCG@K and regression losses are used as proxies for business metrics (CTR, conversion, revenue).  
- The pricing component focuses on predictive modeling and rule-based decision logic, without real-time feedback loops or reinforcement learning.  
- System scalability, A/B testing infrastructure, and large-scale distributed training are outside the current scope.  

Despite these limitations, Bazar reflects realistic ML engineering workflows and architectural decisions used in production e-commerce systems.

---

## Tech Stack

- Python  
- pandas / NumPy  
- scikit-learn  
- CatBoost  
- PyTorch  
- FastAPI  
- MLflow  
- Docker  

---

## Future Improvements

- Online inference service  
- Automated experiment tracking dashboard  
- Feature store abstraction  
- Monitoring & data drift detection  
- Simulation of A/B evaluation framework

---

## Docs
- Architecture overview: `docs/README_ARCH.md`
