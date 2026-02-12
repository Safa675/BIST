# BIST Daily Portfolio Dashboard

## Overview
Daily portfolio monitoring app built on:
- `frontend`: Next.js dashboard UI
- `backend`: FastAPI serving `bist-quant-ai/dashboard/dashboard_data.json`

The app is now dashboard-only (no onboarding or investment-assistant flow).

## Tech Stack
- Frontend: Next.js 14 + TailwindCSS + Framer Motion
- Backend: FastAPI + Pydantic
- Data source: `bist-quant-ai/dashboard/generate_dashboard_data.py`

## Frontend Routes
- `/` → redirects to `/dashboard`
- `/dashboard` → full daily portfolio dashboard

## Backend Endpoints
- `GET /api/health`
- `GET /api/dashboard`
- `GET /api/regime`
- `GET /api/signals`
- `GET /api/signals/{signal_name}`
- `GET /api/portfolio/holdings`
- `GET /api/portfolio/trades`
- `GET /api/portfolio/daily`
- `GET /api/portfolio/summary`
- `GET /api/stats`

## Run
1. Backend:
   `uvicorn main:app --reload --port 8000`
   (from `bist-quant-ai/dashboard/app/backend`)
2. Frontend:
   `npm run dev`
   (from `bist-quant-ai/dashboard/app/frontend`)

## Data Refresh
Regenerate dashboard payload files:
`python bist-quant-ai/dashboard/generate_dashboard_data.py`
