# Player Analysis Platform (Frontend MVP)

Dashboard with team sidebar, video upload, pitch calibration, and heat map. Connects to the FastAPI legacy tracking API (`POST /api/analyze`).

## Features

- Click the **video upload** area to open the OS file picker
- Accepts **MP4** and **MOV** files only (MIME + extension validation)
- Shows selected filename and local video preview
- **API status** in the header (health check against the backend)
- **Analyze** sends the video and player details to `POST /api/analyze`

## Getting started

### Frontend

```bash
npm install
cp .env.local.example .env.local   # optional; defaults to http://localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Backend

From the project root (sibling to `src/`):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
source backend/.venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

Health check: [http://localhost:8000/health](http://localhost:8000/health)

See [backend/README.md](backend/README.md) for pipeline details, env vars, and API schema.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | FastAPI base URL |

Copy `.env.local.example` to `.env.local` to override (`.env.local` is gitignored).

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Production build |
| `npm run start` | Run production server |
| `npm run lint` | Run ESLint |

## Stack

- Next.js 15 (App Router)
- React 19
- TypeScript
- CSS Modules
- FastAPI backend (Python)
