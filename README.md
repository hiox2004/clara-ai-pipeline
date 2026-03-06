# Clara AI Pipeline

Processes demo call transcripts and onboarding call transcripts into versioned account memos and Retell-compatible agent specs.

**Flow:**
```
demo_<id>.txt        →  Pipeline A (n8n)  →  outputs/accounts/<id>/v1/
onboarding_<id>.txt  →  Pipeline B (n8n)  →  outputs/accounts/<id>/v2/  +  changelog
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Ollama](https://ollama.ai) with `llama3` pulled — runs locally, zero cost

```bash
ollama pull llama3
```

---

## Setup

```bash
git clone https://github.com/hiox2004/clara-ai-pipeline
cd clara-ai-pipeline
docker compose up -d
```

Open **http://localhost:5678**

---

## Import Workflows

1. In n8n: **Workflows → Import from file**
2. Import `workflows/pipeline_a.json`  *(Demo call → v1)*
3. Import `workflows/pipeline_b.json`  *(Onboarding call → v2 + changelog)*

---

## How to Process a Client

### Step 1 — Add transcripts

Drop transcript files into the correct folders:

```
data/transcripts/demo/demo_<id>.txt
data/transcripts/onboarding/onboarding_<id>.txt
```

The `<id>` suffix must match between the two files.
**Example:** `demo_claraai.txt` pairs with `onboarding_claraai.txt`

### Step 2 — Run Pipeline A (Demo → v1)

In n8n, open **Pipeline A** and click **Execute workflow**.

Reads every `.txt` file in `data/transcripts/demo/` and writes:
```
outputs/accounts/<id>/v1/account_memo_v1.json
outputs/accounts/<id>/v1/agent_spec_v1.json
```

### Step 3 — Run Pipeline B (Onboarding → v2)

In n8n, open **Pipeline B** and click **Execute workflow**.

Reads every `.txt` file in `data/transcripts/onboarding/`, patches the v1 memo, and writes:
```
outputs/accounts/<id>/v2/account_memo_v2.json
outputs/accounts/<id>/v2/agent_spec_v2.json
outputs/accounts/<id>/v2/changes.json
outputs/changelog/<id>_changes.json
```

> **Pipeline B requires Pipeline A outputs to exist first.**

---

## Transcript Naming

| Demo file | Onboarding file | account_id used |
|-----------|----------------|-----------------|
| `demo_claraai.txt` | `onboarding_claraai.txt` | `demo_claraai` |
| `demo_001.txt` | `onboarding_001.txt` | `demo_001` |
| `demo_acme.txt` | `onboarding_acme.txt` | `demo_acme` |

Any `.txt` filename works as long as the suffixes match.

---

## Output Files

### `account_memo_v{n}.json`
Structured account profile extracted from the transcript — hours, services, emergency routing, integrations, etc.

### `agent_spec_v{n}.json`
Retell-ready agent configuration including `system_prompt`, `call_transfer_protocol`, `key_variables`, and `tool_invocation_placeholders`.

### `changes.json`
Field-level diff of what changed between v1 and v2.

---

## Alternative: Python CLI (no Docker required)

```bash
# All accounts (auto-discovers transcript files)
python scripts/run_pipeline.py --no-ollama

# Single account
python scripts/run_pipeline.py --account demo_claraai --no-ollama

# Pipeline A only
python scripts/run_pipeline.py --phase a --no-ollama
```

Requires Python 3.9+. No pip installs. Ollama optional (rule-based fallback if not running).

---

## Retell Integration

The `agent_spec_v2.json` `system_prompt` field is paste-ready into the Retell agent dashboard. With an API key, the spec maps directly to Retell's `POST /create-agent` endpoint.
