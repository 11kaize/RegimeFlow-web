---
title: Regimeflow
emoji: 📉
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🧬 RegimeFlow — Biological Trajectory Forecasting

Interactive web demo for ICML 2026 paper "A Regime-Aware Trajectory Prediction Framework for 1000+ Systems Biology Models".

## Features

- **Force-directed graph** — Explore 14 time-series models and their relationships
- **Trajectory prediction** — Select biological systems, predict future 256 steps with Chronos-Bolt

## Tech Stack

- **Backend**: FastAPI + Chronos-Bolt (Amazon's zero-shot forecaster)
- **Frontend**: Vanilla HTML/CSS/JS + ECharts
- **Deployment**: Docker SDK (16GB free RAM)

## Endpoints

- `/` — Web demo
- `/api/health` — Health check
- `/api/predict` — Single-species trajectory prediction
- `/api/predict/multi` — Multi-species batch prediction
