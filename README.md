# Antiplagiarism FastAPI Service

A local-first MVP for plagiarism detection. The app ingests PDF documents, extracts text, runs OCR on scanned pages, stores a searchable corpus in PostgreSQL, and compares new documents against the database with multilingual embeddings.

## Features

- PDF document ingestion with metadata storage
- Background indexing for uploaded documents
- OCR fallback for image-based or empty pages
- DB-backed plagiarism checks with chunk-level matches
- Persisted check reports and match details
- Health endpoints for basic app and readiness checks

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [PostgreSQL](https://www.postgresql.org/) with [pgvector](https://github.com/pgvector/pgvector)
- [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- [Tesseract trained data files](https://tesseract-ocr.github.io/tessdoc/Data-Files.html)
- [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [pytesseract](https://github.com/madmaze/pytesseract)
- `ibm-granite/granite-embedding-311m-multilingual-r2`

## Requirements

- Python 3.11+
- PostgreSQL with the `pgvector` extension enabled
- Tesseract OCR
- Access to the embedding model above, either through the local Hugging Face cache or an internet connection on first run

### Tesseract notes

- On Windows, the recommended path is the Tesseract installer from [UB Mannheim](https://ub-mannheim.github.io/Tesseract_Dokumentation/Tesseract_Doku_Windows.html).
- The official Tesseract docs point to the trained data repositories and language packages. `tessdata_fast` is a good default for most use cases.
- If `tesseract.exe` is not on your `PATH`, set `TESSERACT_CMD` in `.env`.
- If your `tessdata` folder is not in the default location, set `TESSDATA_PREFIX` in your shell or service config.
- On Windows, use forward slashes in `.env` to avoid escaping issues, for example:
  `TESSERACT_CMD="C:/Program Files/Tesseract-OCR/tesseract.exe"`

## Quick Start

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate it:
   ```bash
   # Windows
   .venv\Scripts\activate

   # Linux / macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and set at least:
   - `DATABASE_URL`
   - `MODEL_ID`
   - `TESSERACT_CMD` if Tesseract is not in `PATH`
   - `UPLOAD_DIR` if you want to store files somewhere other than the default

5. Enable the PostgreSQL extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

6. Run database migrations:
   ```bash
   alembic upgrade head
   ```

7. Start the API:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

   If port `8000` is busy, add `--port 8001` or another free port.

## API

- `GET /health`
- `GET /api/v1/health`
- `GET /docs`
- `POST /api/v1/documents`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}`
- `DELETE /api/v1/documents/{document_id}`
- `POST /api/v1/checks`
- `GET /api/v1/checks/{check_id}`
- `GET /api/v1/checks/{check_id}/matches`

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Async SQLAlchemy connection string |
| `MODEL_ID` | Embedding model identifier |
| `UPLOAD_DIR` | Directory for uploaded PDFs |
| `TESSERACT_CMD` | Full path to `tesseract.exe` if it is not on `PATH` |
| `MATCH_THRESHOLD` | Minimum similarity threshold for a suspicious match |
| `SUSPECT_OVERLAP` | Minimum overlap used in scoring |
| `SKIP_INTRO_CHUNKS` | Leading chunks ignored when ranking matches |
| `TOP_K_MATCHES` | Number of top chunk matches kept per result |

## Project Layout

- `app/` - FastAPI application, services, repositories, and schemas
- `tests/` - Pytest suite
- `alembic/` - Database migrations
- `raw_jupyter_code/` - Legacy notebooks and source code
- `requirements.txt` - Python dependencies

## Useful Links

- [pgvector](https://github.com/pgvector/pgvector)
- [Tesseract installation docs](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- [Tesseract data files](https://tesseract-ocr.github.io/tessdoc/Data-Files.html)
- [Tesseract Windows installer info](https://ub-mannheim.github.io/Tesseract_Dokumentation/Tesseract_Doku_Windows.html)
- [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast)

## Notes

- Uploaded documents are indexed in the background after the API responds.
- Check results are persisted in PostgreSQL and can be retrieved later by ID.
- The OCR pipeline is designed to work with scanned or partially empty PDF pages.
