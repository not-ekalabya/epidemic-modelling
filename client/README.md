## Client

This Next.js app now includes a thin backend route at `app/api/predict/route.ts`.
It proxies browser requests to the Python prediction API.

### Run

1. Start the Python API from the repo root:
```bash
uvicorn server.server:app --host 0.0.0.0 --port 8000
```

2. Start the client:
```bash
cd client
npm run dev
```

3. Open `http://localhost:3000`

### Backend URL

By default, the client proxy forwards to `http://127.0.0.1:8000`.

To change that, set:

```bash
PREDICTION_API_BASE_URL=http://127.0.0.1:8000
```

### Flow

- Browser submits to `POST /api/predict`
- Payload includes `prediction_date` and optional `include_actual`
- Next route handler forwards the payload to the FastAPI backend
- Response contains country-level predicted and actual trajectories
- Results are rendered on the home page
