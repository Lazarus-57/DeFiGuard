React frontend scaffold (Phase 2) — instructions and sample files.

Goal
----
Provide a small React app that uploads a CSV of transactions to the FastAPI backend and displays model results.

How to create the React app (recommended)
---------------------------------------
1. From the DeFIGuard folder run:

   npx create-react-app frontend
   cd frontend
   npm install axios

2. Replace src/App.js with the sample in this repo (see files in the `DeFIGuard/Frontend` folder) and start the dev server:

   npm start

3. The React app will POST CSV files to http://localhost:8000/predict (FastAPI server must be running).

What to copy from this repository
--------------------------------
- Frontend/frontend_App.js.sample  -> copy into frontend/src/App.js
- Frontend/frontend_index_js.sample -> copy into frontend/src/index.js
- Frontend/frontend_public_index_html.sample -> copy into frontend/public/index.html (optional)

Notes
-----
- The FastAPI server is at DeFIGuard/api.py. Run it with:

    python -m uvicorn api:app --reload --port 8000

- Ensure Python virtual environment has the Phase 1 dependencies plus FastAPI/uvicorn/python-multipart (requirements.txt updated).

- This scaffold keeps the React app separate so frontend dependencies are managed with npm (not pip).

Files included in this repo as samples (copy into the frontend created by create-react-app):
- Frontend/frontend_App.js.sample
- Frontend/frontend_index_js.sample
- Frontend/frontend_public_index_html.sample
