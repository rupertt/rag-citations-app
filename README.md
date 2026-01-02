# rag-citations-app

Minimal local **Python 3.11** RAG web app that answers questions using **OpenAI + LangChain** with **citations** from a single hardcoded file: `./data/doc.txt`.

## What it does

- Reads `./data/doc.txt`
- Chunks it into stable chunk IDs: `chunk-00`, `chunk-01`, ...
- Embeds + indexes into **local persistent Chroma** at `./data/index` (so you don’t re-embed every run)
- Retrieves top_k chunks
- Answers using only retrieved chunks and includes citations like `[doc.txt#chunk-03]`

If the documentation doesn’t contain enough info, the API returns:

`I can’t find that in the provided documentation.`

## Setup (Windows)

From PowerShell:

```powershell
cd "C:\Users\Rupert\Desktop\Coding Projects\Agentic AI\rag-citations-app"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

Run:

```powershell
uvicorn app.main:app --reload
```

## Setup (Linux / WSL)

```bash
cd "rag-citations-app"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

Run:

```bash
uvicorn app.main:app --reload
```

## API

### Health

- `GET /health` → `{"status":"ok"}`

### Ask

- `POST /ask`

Request:

```json
{"question":"<string>","top_k":4,"debug":false}
```

Example curl:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What does this service do?","top_k":4,"debug":true}'
```

Response shape:

```json
{
  "answer": "<string>",
  "citations": [{"source":"doc.txt","chunk_id":"chunk-03","snippet":"..."}],
  "debug": {
    "retrieved": [{"chunk_id":"chunk-03","text":"...","score":0.12}]
  }
}
```


