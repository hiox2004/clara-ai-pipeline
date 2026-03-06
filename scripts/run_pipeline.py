#!/usr/bin/env python3
"""
Clara AI Pipeline - Batch Processing Script
============================================
Drops any demo + onboarding transcripts into the data/transcripts folders
and this script pairs them, runs both pipelines, and writes versioned JSON outputs.

Usage:
  python scripts/run_pipeline.py              # process all discovered accounts
  python scripts/run_pipeline.py --account demo_claraai
  python scripts/run_pipeline.py --phase a   # only pipeline A (demo -> v1)
  python scripts/run_pipeline.py --phase b   # only pipeline B (onboarding -> v2)

Transcript naming convention:
  data/transcripts/demo/demo_<id>.txt
  data/transcripts/onboarding/onboarding_<id>.txt
  The <id> suffix must match between the two files.

LLM: Uses Ollama (llama3) at http://localhost:11434 if available.
     Falls back to rule-based extraction if Ollama is not running.
     Zero-cost, no paid APIs required.
"""

import argparse
import json
import os
import re
import sys
import datetime
import urllib.request
import urllib.error

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(ROOT_DIR, "data", "transcripts", "demo")
ONBOARDING_DIR = os.path.join(ROOT_DIR, "data", "transcripts", "onboarding")


def discover_accounts() -> dict:
    """Auto-discover account pairs from the transcript folders.

    Pairs files by matching the <id> suffix:
      demo/demo_<id>.txt  <-->  onboarding/onboarding_<id>.txt

    Returns a dict keyed by account_id with demo_file and onboarding_file paths.
    """
    accounts = {}
    for fname in sorted(os.listdir(DEMO_DIR)):
        if not fname.endswith(".txt") or fname == ".gitkeep":
            continue
        stem = os.path.splitext(fname)[0]           # e.g. "demo_claraai", "demo_001"
        account_id = stem                            # use full stem as account ID

        # Derive onboarding filename: replace leading "demo_" with "onboarding_"
        if stem.startswith("demo_"):
            suffix = stem[len("demo_"):]
            onboarding_fname = f"onboarding_{suffix}.txt"
        else:
            onboarding_fname = f"onboarding_{stem}.txt"

        onboarding_path = os.path.join(ONBOARDING_DIR, onboarding_fname)
        if not os.path.exists(onboarding_path):
            print(f"  ⚠  No matching onboarding file for {fname} (expected {onboarding_fname}) — Pipeline A only.")
            onboarding_path = None

        accounts[account_id] = {
            "demo_file": os.path.join("data", "transcripts", "demo", fname),
            "onboarding_file": os.path.join("data", "transcripts", "onboarding", onboarding_fname)
                               if onboarding_path else None,
        }
    return accounts

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

MEMO_SCHEMA = {
    "account_id": "",
    "company_name": "",
    "business_hours": {"days": "", "start": "", "end": "", "timezone": ""},
    "office_address": "",
    "services_supported": [],
    "emergency_definition": [],
    "emergency_routing_rules": {"primary": "", "order": [], "fallback": ""},
    "non_emergency_routing_rules": "",
    "call_transfer_rules": {"timeout_seconds": None, "retries": None, "message_if_fails": ""},
    "integration_constraints": [],
    "after_hours_flow_summary": "",
    "office_hours_flow_summary": "",
    "questions_or_unknowns": [],
    "notes": "",
}

# ── Ollama helpers ─────────────────────────────────────────────────────────────

def ollama_available():
    try:
        req = urllib.request.Request("http://localhost:11434/", method="GET")
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def ollama_extract(prompt: str) -> dict | None:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": False,
        "prompt": prompt,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            raw = result.get("response", "")
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                return json.loads(m.group(0))
    except Exception as e:
        print(f"  [Ollama] Error: {e}")
    return None


# ── Rule-based extraction ──────────────────────────────────────────────────────

# Known non-client speaker names (Clara sales / onboarding team members)
_CLARA_SPEAKERS = {"agent", "clara", "bharat", "nick", "pavan", "naveen", "speaker"}


def _client_text(transcript: str) -> str:
    """Return concatenated client-turn text.

    Handles three formats:
    - Structured:         Agent: / Client: prefixes
    - Named speakers:     Ben: / Bharat: / Nick:
    - Numbered speakers:  Speaker 2 (Ben): / Speaker 1 (Bharat):
    """
    lines = []

    # -- Try structured Agent:/Client: format first --
    if any(l.startswith("Client:") for l in transcript.splitlines()):
        in_client = False
        for line in transcript.splitlines():
            if line.startswith("Client:"):
                in_client = True
                lines.append(line[7:].strip())
            elif line.startswith("Agent:"):
                in_client = False
            elif in_client and line.strip():
                lines.append(line.strip())
        return " ".join(lines)

    # -- Try 'Speaker N (Name):' format --
    numbered_re = re.compile(r"^Speaker\s+\d+\s+\(([A-Za-z]+)\):\s*(.*)")
    numbered_speakers: dict[str, int] = {}
    for line in transcript.splitlines():
        m = numbered_re.match(line)
        if m:
            name = m.group(1).strip().lower()
            numbered_speakers[name] = numbered_speakers.get(name, 0) + 1

    if numbered_speakers:
        # Client = non-Clara speaker with most turns
        client_name = ""
        best_count = 0
        for name, count in numbered_speakers.items():
            if name not in _CLARA_SPEAKERS and count > best_count:
                best_count = count
                client_name = name
        if client_name:
            prefix = f"({client_name.title()}):"
            for line in transcript.splitlines():
                if prefix.lower() in line.lower():
                    m = numbered_re.match(line)
                    if m:
                        lines.append(m.group(2).strip())
            return " ".join(lines)

    # -- Plain named-speaker format: Ben: / Bharat: etc. --
    speaker_re = re.compile(r"^([A-Za-z][A-Za-z]{1,20}):\s")
    all_speakers: dict[str, int] = {}
    for line in transcript.splitlines():
        m = speaker_re.match(line)
        if m:
            name = m.group(1).strip().lower()
            all_speakers[name] = all_speakers.get(name, 0) + 1

    client_name = ""
    best_count = 0
    for name, count in all_speakers.items():
        if name not in _CLARA_SPEAKERS and count > best_count:
            best_count = count
            client_name = name

    if client_name:
        prefix = client_name + ":"
        for line in transcript.splitlines():
            if line.lower().startswith(prefix.lower()):
                lines.append(line[len(prefix):].strip())
        return " ".join(lines)

    # Fallback: return full transcript
    return transcript


def _find_company(text: str) -> str:
    # 1. Highest confidence: extract from business email domain
    #    e.g. info@benselectricsolutionsteam.com → "Bens Electric Solutions"
    email_m = re.search(r"\b[a-z]+@([a-z0-9]{6,})\.com", text, re.IGNORECASE)
    if email_m:
        slug = email_m.group(1)
        # Strip trailing domain suffixes (not part of company name)
        slug = re.sub(r"(?:team|corp|inc|co|llc|ltd)$", "", slug, flags=re.IGNORECASE)
        # Insert spaces before known industry words
        for word in ["solutions", "services", "electric", "electrical", "plumbing",
                     "hvac", "roofing", "exteriors", "protection", "dispatch", "flow"]:
            slug = re.sub(r"(?<=[a-z])" + word, " " + word, slug, flags=re.IGNORECASE)
        result = slug.strip().title()
        if len(result) > 3:
            return result

    # 2. Explicit self-identification patterns
    intro_patterns = [
        r"(?:We are|We're|I'm \w+ from|I am \w+ from)\s+([A-Z][A-Za-z0-9 &'\.]+?)(?:\s+(?:based|in|located|\.)|,)",
        r"([A-Z][A-Za-z0-9 &']+(?:HVAC|Plumbing|Roofing|Electric(?:al)?(?:\s+Solutions)?|Fire\s+Protection|Exteriors?))",
        r"(?:my (?:company|business) (?:is called|is named|is)?)\s+([A-Z][A-Za-z0-9 &'\.]{3,40}?)[.,]",
    ]
    for p in intro_patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip().rstrip(".")
    return ""


def _find_hours(text: str) -> dict:
    bh = {"days": "", "start": "", "end": "", "timezone": ""}
    # Days (check longer spans first)
    if re.search(r"Monday\s+(?:to|through)\s+Saturday", text, re.IGNORECASE):
        bh["days"] = "Monday to Saturday"
    elif re.search(r"Monday\s+(?:to|through)\s+Friday.*?Saturday", text, re.IGNORECASE | re.DOTALL):
        bh["days"] = "Monday to Saturday"
    elif re.search(r"Monday\s+(?:to|through)\s+Friday", text, re.IGNORECASE):
        bh["days"] = "Monday to Friday"
    elif re.search(r"Monday\s+(?:to|through)\s+Sunday|7\s+days", text, re.IGNORECASE):
        bh["days"] = "Monday to Sunday"
    # Time range — match with or without am/pm (e.g. "8 to 4:30" or "7am to 6pm")
    # Require at least one side has am/pm OR both sides are plausible hour values (1-12)
    m = re.search(r"(\d+(?::\d+)?(?:am|pm)?)\s+to\s+(\d+(?::\d+)?(?:am|pm)?)", text, re.IGNORECASE)
    if m:
        s, e = m.group(1), m.group(2)
        # Accept if either has am/pm, or both look like clock hours (1-12)
        def _is_hour(t: str) -> bool:
            base = int(re.match(r"\d+", t).group())
            return 1 <= base <= 12
        if re.search(r"am|pm", s + e, re.IGNORECASE) or (_is_hour(s) and _is_hour(e)):
            bh["start"] = s.lower()
            bh["end"] = e.lower()
    # Timezone
    for tz in ["Eastern", "Central", "Mountain", "Pacific"]:
        if re.search(r"\b" + tz + r"\b", text, re.IGNORECASE):
            bh["timezone"] = tz
            break
    return bh


def _find_services(text: str) -> list:
    # Look for "We do/handle/offer X, Y, and Z" clause
    # Exclude fee/pricing statements
    m = re.search(
        r"(?:We do|We handle|We offer|We provide|we do)\s+([^.]+?)(?:\.|Our office|$)",
        text, re.IGNORECASE
    )
    if m:
        raw = m.group(1)
        # Skip if the match is about fees/pricing
        if re.search(r"fee|charge|cost|hour|price|\$", raw, re.IGNORECASE):
            return []
        parts = re.split(r",\s*(?:and\s*)?|\s+and\s+", raw)
        cleaned = []
        for p in parts:
            p = p.strip().rstrip(",. ")
            # Remove leading verbs/articles
            p = re.sub(r"^(?:we\s+(?:sell|do|also\s+do|offer|provide|handle)\s+)", "", p, flags=re.IGNORECASE)
            if p and len(p) < 60:  # Skip overly long matches
                cleaned.append(p)
        return cleaned
    return []


def _find_emergency_routing(text: str) -> dict:
    routing = {"primary": "", "order": [], "fallback": ""}

    # Primary contact
    primary_m = re.search(
        r"(?:first try|route to|call our emergency line at|dispatcher|contact)\s+"
        r"(?:([A-Z][a-z]+ [A-Z][a-z]+)\s+at\s+)?(\d{3}-\d{3}-\d{4})",
        text, re.IGNORECASE,
    )
    if primary_m:
        name, number = primary_m.group(1), primary_m.group(2)
        entry = f"{name} at {number}" if name else number
        routing["primary"] = entry
        routing["order"].append(entry)

    # Secondary contact — require "no answer" context to avoid re-matching primary
    sec_m = re.search(
        r"(?:no (?:answer|pickup).*?try|then try|try (?:the )?backup|second(?:ary)?\s+(?:try|line|contact))\s+"
        r"(?:[a-z ]*?\s*)?(?:([A-Z][a-z]+ [A-Z][a-z]+)\s+at\s+)?(\d{3}-\d{3}-\d{4})",
        text, re.IGNORECASE | re.DOTALL,
    )
    if sec_m:
        name, number = sec_m.group(1), sec_m.group(2)
        entry = f"{name} at {number}" if name else number
        if entry not in routing["order"]:
            routing["order"].append(entry)

    # Fallback email / SMS target
    fb_m = re.search(
        r"(?:fail[s]?|both fail)\s+(?:send|email|SMS|alert)\s+(?:[a-z ]+\s+)?([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
        text, re.IGNORECASE,
    )
    if fb_m:
        routing["fallback"] = fb_m.group(1)

    return routing


def _find_transfer_rules(text: str) -> dict:
    rules = {"timeout_seconds": None, "retries": None, "message_if_fails": ""}
    t_m = re.search(r"(\d+)\s+seconds", text, re.IGNORECASE)
    if t_m:
        rules["timeout_seconds"] = int(t_m.group(1))
    if re.search(r"no retries", text, re.IGNORECASE):
        rules["retries"] = 0
    elif re.search(r"one retry", text, re.IGNORECASE):
        rules["retries"] = 1
    # callback promise
    cb_m = re.search(r"(?:call(?:back| back)|call them back)(.*?)(?:\.|$)", text, re.IGNORECASE)
    if cb_m:
        rules["message_if_fails"] = "Apologize and assure callback. " + cb_m.group(0).strip()
    return rules


def _find_integrations(text: str) -> list:
    found = []
    for name in ["ServiceTitan", "Housecall Pro", "JobNimbus", "Jobber", "ServiceTrade"]:
        if name.lower() in text.lower():
            found.append(name)
    return found


def rule_based_extract(transcript: str, account_id: str, source: str) -> dict:
    """Deterministic extraction from structured Agent/Client transcript."""
    memo = json.loads(json.dumps(MEMO_SCHEMA))  # deep copy
    memo["account_id"] = account_id

    client_text = _client_text(transcript)
    full = transcript

    # Helper: try client-only text first, fall back to full transcript if empty
    def _try(fn, *args):
        result = fn(client_text, *args)
        if not result:
            result = fn(full, *args)
        return result

    memo["company_name"] = _try(_find_company)
    memo["business_hours"] = _find_hours(client_text) if _find_hours(client_text)["start"] else _find_hours(full)

    # Office address (city, state)
    for search_text in (client_text, full):
        loc_m = re.search(r"(?:based in|in|located in)\s+([A-Z][A-Za-z ]+,\s*[A-Z]{2})", search_text)
        if loc_m:
            memo["office_address"] = loc_m.group(1).strip()
            break

    memo["services_supported"] = _try(_find_services)
    memo["emergency_routing_rules"] = _find_emergency_routing(client_text)
    if not memo["emergency_routing_rules"]["primary"]:
        memo["emergency_routing_rules"] = _find_emergency_routing(full)
    memo["call_transfer_rules"] = _find_transfer_rules(client_text) if _find_transfer_rules(client_text)["timeout_seconds"] else _find_transfer_rules(full)
    memo["integration_constraints"] = _try(_find_integrations)

    # Non-emergency routing
    ne_m = None
    for search_text in (client_text, full):
        ne_m = re.search(
            r"(?:non[- ]emergency|after[- ]hours.*?(?:collect|take))\s*[:\-]?\s*(.+?)(?:\.|$)",
            search_text, re.IGNORECASE
        )
        if ne_m:
            break
    if ne_m:
        memo["non_emergency_routing_rules"] = ne_m.group(1).strip()
    else:
        memo["non_emergency_routing_rules"] = "Collect name, number, and issue description. Follow up next business day."

    # Emergency definition — search client text first, then full transcript
    for search_text in (client_text, full):
        emerg_m = re.search(
            r"(?:No\s+heat|AC\s+(?:failure|failures|down|out)|burst\s+pipe|gas\s+leak|sewage\s+backup|carbon\s+monoxide|storm\s+damage|sprinkler\s+leak|fire\s+alarm|flood|power\s+out)[^.]*",
            search_text, re.IGNORECASE,
        )
        if emerg_m:
            raw = emerg_m.group(0)
            parts = re.split(r",\s*(?:or\s*)?|\s+or\s+", raw)
            memo["emergency_definition"] = [p.strip().rstrip(",. ") for p in parts if p.strip()]
            break
    if not memo["emergency_definition"]:
        for search_text in (client_text, full):
            emerg_m2 = re.search(
                r"(?:emergency|emergencies)\s+(?:is|are)[^.]*?(?:is|are)\s+(.+?)(?:\.|only|$)",
                search_text, re.IGNORECASE,
            )
            if emerg_m2:
                raw = emerg_m2.group(1)
                memo["emergency_definition"] = [e.strip() for e in re.split(r",\s*(?:and\s*)?|\s+and\s+", raw) if e.strip()]
                break

    # Summaries
    bh = memo["business_hours"]
    if bh["days"] and bh["start"]:
        memo["office_hours_flow_summary"] = (
            f"Greet caller, identify purpose, collect name and number, transfer to team. "
            f"Hours: {bh['days']}, {bh['start']} to {bh['end']} {bh['timezone']}."
        )
    memo["after_hours_flow_summary"] = (
        "Greet caller, identify purpose, confirm if emergency. "
        "If emergency: collect name/number/address and attempt transfer. "
        "If non-emergency: collect details and confirm next-day follow-up."
    )

    memo["version"] = "v1"
    memo["source"] = source
    return memo


# ── LLM extraction (Ollama) ───────────────────────────────────────────────────

DEMO_PROMPT_TEMPLATE = """You are an expert at extracting structured business information from call transcripts.

Extract the following fields from this transcript and return ONLY valid JSON, no explanation, no markdown:

{{
  "account_id": "{account_id}",
  "company_name": "",
  "business_hours": {{ "days": "", "start": "", "end": "", "timezone": "" }},
  "office_address": "",
  "services_supported": [],
  "emergency_definition": [],
  "emergency_routing_rules": {{ "primary": "", "order": [], "fallback": "" }},
  "non_emergency_routing_rules": "",
  "call_transfer_rules": {{ "timeout_seconds": null, "retries": null, "message_if_fails": "" }},
  "integration_constraints": [],
  "after_hours_flow_summary": "",
  "office_hours_flow_summary": "",
  "questions_or_unknowns": [],
  "notes": ""
}}

Rules:
- Only extract what is EXPLICITLY stated. Do not guess or invent.
- If a field is missing, leave it blank or empty array.
- Put anything unclear in questions_or_unknowns.
- Return ONLY the JSON object, nothing else.

Transcript:
{transcript}"""


def extract_memo(transcript: str, account_id: str, source: str, use_ollama: bool) -> dict:
    """Extract account memo using Ollama if available, else rule-based."""
    if use_ollama:
        print(f"  [LLM] Calling Ollama for {account_id} ({source})...")
        prompt = DEMO_PROMPT_TEMPLATE.format(account_id=account_id, transcript=transcript)
        result = ollama_extract(prompt)
        if result:
            result.setdefault("account_id", account_id)
            result["version"] = "v1"
            result["source"] = source
            # Ensure all required keys exist
            for k, v in MEMO_SCHEMA.items():
                if k not in result:
                    result[k] = v
            return result
        print(f"  [LLM] Ollama returned no result, falling back to rule-based.")
    print(f"  [Rule] Using rule-based extraction for {account_id} ({source})...")
    return rule_based_extract(transcript, account_id, source)


# ── Onboarding patch (v1 → v2) ────────────────────────────────────────────────

def _diff(old, new, path="") -> list:
    """Recursively compare two dicts and return list of change records."""
    changes = []
    all_keys = set(list(old.keys()) + list(new.keys()))
    for k in all_keys:
        old_v = old.get(k)
        new_v = new.get(k)
        if new_v is None or new_v == "" or new_v == [] or new_v == {}:
            continue
        if isinstance(new_v, dict) and isinstance(old_v, dict):
            changes.extend(_diff(old_v, new_v, path + k + "."))
        elif json.dumps(old_v, sort_keys=True) != json.dumps(new_v, sort_keys=True):
            changes.append({"field": path + k, "from": old_v, "to": new_v})
    return changes


def _apply_patch(base: dict, updates: dict) -> dict:
    """Deep-merge updates into base (skip empty values)."""
    result = json.loads(json.dumps(base))
    for k, v in updates.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _apply_patch(result[k], v)
        else:
            result[k] = v
    return result


def build_v2(v1_memo: dict, onboarding_transcript: str, account_id: str, use_ollama: bool) -> tuple[dict, list]:
    """Extract onboarding updates, patch v1, return (v2_memo, changes)."""
    updates = extract_memo(onboarding_transcript, account_id, "onboarding_call", use_ollama)

    # Strip meta fields before diffing
    for field in ("version", "source", "account_id"):
        updates.pop(field, None)

    changes = _diff(v1_memo, updates)
    v2 = _apply_patch(v1_memo, updates)
    v2["version"] = "v2"
    v2["source"] = "onboarding_call"
    return v2, changes


# ── Agent Spec generator ──────────────────────────────────────────────────────

def generate_spec(memo: dict) -> dict:
    bh = memo.get("business_hours") or {}
    er = memo.get("emergency_routing_rules") or {}
    transfer = memo.get("call_transfer_rules") or {}
    emergency_def = memo.get("emergency_definition") or []
    constraints = memo.get("integration_constraints") or []
    company = memo.get("company_name") or memo["account_id"]
    version = memo.get("version", "v1")
    timeout = transfer.get("timeout_seconds") or 60

    def hours_str():
        if bh.get("days") and bh.get("start"):
            return f"{bh['days']}, {bh['start']}-{bh['end']} {bh.get('timezone', '')}".strip()
        return "Mon-Fri, 8AM-5PM"

    def er_order_str():
        order = er.get("order") or []
        return json.dumps(order)

    prompt_lines = [
        f'You are Clara, an AI receptionist for {company}.\n',
        f'BUSINESS HOURS FLOW ({hours_str()}):',
        f'1. Greet: "Thank you for calling {company}, this is Clara. How can I help you today?"',
        '2. Listen to their purpose.',
        '3. Collect their name and callback number.',
        '4. Transfer to the appropriate team.',
        f'5. If transfer fails after {timeout} seconds: "I\'m sorry I wasn\'t able to connect you directly. I\'ve noted your information and someone will call you back shortly. Is there anything else I can help you with?"',
        '6. Ask if they need anything else, then close.\n',
        'AFTER HOURS FLOW:',
        f'1. Greet: "Thank you for calling {company}. You\'ve reached us outside of business hours."',
        '2. Ask the purpose of their call.',
        '3. Ask: "Is this an emergency?"',
    ]

    if emergency_def:
        emerg_str = ", ".join(emergency_def)
        prompt_lines += [
            f'4. IF EMERGENCY ({emerg_str}):',
            '   - Collect name, callback number, and site address immediately.',
            f'   - Attempt transfer to: {er.get("primary") or "on-call technician"}.',
            f'   - Fallback order: {er_order_str()}.',
            '   - If all transfers fail: "I wasn\'t able to reach the on-call team. I\'ve captured your details and dispatched an alert. Someone will reach you very shortly."',
        ]
    else:
        prompt_lines += [
            f'4. IF EMERGENCY:',
            '   - Collect name, callback number, and site address immediately.',
            f'   - Attempt transfer to: {er.get("primary") or "on-call technician"}.',
            '   - If transfer fails: "I\'ve captured your details and someone will reach you very shortly."',
        ]

    prompt_lines += [
        '5. IF NON-EMERGENCY:',
        '   - Collect name, number, and brief description.',
        '   - Confirm: "I\'ve noted your request and someone will follow up during business hours."',
        '6. Ask if they need anything else, then close.\n',
        'RULES:',
        '- Never mention function calls or tools to the caller.',
        '- Only collect what is needed for routing.',
        '- Do not ask more questions than necessary.',
        '- Transfer protocol: attempt transfer, wait, fallback to message if no answer.',
    ]

    if constraints:
        prompt_lines.append(f'- Constraints: {"; ".join(constraints)}')

    spec = {
        "account_id": memo["account_id"],
        "agent_name": f"Clara - {company}",
        "voice_style": "professional, calm, empathetic",
        "system_prompt": "\n".join(prompt_lines),
        "key_variables": {
            "timezone": bh.get("timezone", ""),
            "business_hours": bh,
            "office_address": memo.get("office_address", ""),
            "emergency_routing": er,
        },
        "tool_invocation_placeholders": ["transfer_call", "create_ticket", "lookup_account"],
        "call_transfer_protocol": transfer,
        "fallback_protocol": "If all transfers fail, apologize, assure callback, log the attempt.",
        "version": version,
    }
    return spec


# ── File I/O ──────────────────────────────────────────────────────────────────

def read_transcript(relative_path: str) -> str:
    full = os.path.join(ROOT_DIR, relative_path)
    with open(full, "r", encoding="utf-8-sig") as f:
        return f.read().strip()


def write_json(relative_path: str, data: dict):
    full = os.path.join(ROOT_DIR, relative_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Written: {relative_path}")


# ── Pipeline A: demo → v1 ─────────────────────────────────────────────────────

def run_pipeline_a(account_id: str, files: dict, use_ollama: bool):
    print(f"\n[Pipeline A] {account_id}")
    transcript = read_transcript(files["demo_file"])
    memo = extract_memo(transcript, account_id, "demo_call", use_ollama)
    spec = generate_spec(memo)

    write_json(f"outputs/accounts/{account_id}/v1/account_memo_v1.json", memo)
    write_json(f"outputs/accounts/{account_id}/v1/agent_spec_v1.json", spec)


# ── Pipeline B: onboarding → v2 + changelog ───────────────────────────────────

def run_pipeline_b(account_id: str, files: dict, use_ollama: bool):
    print(f"\n[Pipeline B] {account_id}")

    # Load v1 memo
    v1_path = os.path.join(ROOT_DIR, f"outputs/accounts/{account_id}/v1/account_memo_v1.json")
    if not os.path.exists(v1_path):
        print(f"  [!] v1 memo not found at {v1_path}. Run Pipeline A first.")
        return

    with open(v1_path, "r") as f:
        v1_memo = json.load(f)

    onboarding_transcript = read_transcript(files["onboarding_file"])
    v2_memo, changes = build_v2(v1_memo, onboarding_transcript, account_id, use_ollama)
    v2_spec = generate_spec(v2_memo)

    changelog = {
        "account_id": account_id,
        "from_version": "v1",
        "to_version": "v2",
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "changes": changes,
    }

    write_json(f"outputs/accounts/{account_id}/v2/account_memo_v2.json", v2_memo)
    write_json(f"outputs/accounts/{account_id}/v2/agent_spec_v2.json", v2_spec)
    write_json(f"outputs/accounts/{account_id}/v2/changes.json", changelog)
    write_json(f"outputs/changelog/{account_id}_changes.json", changelog)
    print(f"  ✓ {len(changes)} change(s) recorded")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clara AI Pipeline batch runner")
    parser.add_argument("--account", default=None, help="Single account ID to process (e.g. demo_claraai)")
    parser.add_argument("--phase", default="both", choices=["a", "b", "both"],
                        help="Which pipeline to run: a (demo→v1), b (onboarding→v2), or both")
    parser.add_argument("--no-ollama", action="store_true", help="Skip Ollama, use rule-based only")
    args = parser.parse_args()

    all_accounts = discover_accounts()
    if not all_accounts:
        print("❌  No demo transcripts found in data/transcripts/demo/")
        print("    Drop demo_<id>.txt files there and matching onboarding_<id>.txt in data/transcripts/onboarding/")
        sys.exit(1)

    if args.account:
        if args.account not in all_accounts:
            print(f"❌  Account '{args.account}' not found. Available: {list(all_accounts.keys())}")
            sys.exit(1)
        accounts_to_run = {args.account: all_accounts[args.account]}
    else:
        accounts_to_run = all_accounts

    if args.no_ollama:
        use_ollama = False
        print("ℹ  Ollama disabled by flag. Using rule-based extraction.")
    else:
        use_ollama = ollama_available()
        if use_ollama:
            print("✓  Ollama detected at localhost:11434 — using LLM extraction.")
        else:
            print("ℹ  Ollama not detected. Using rule-based extraction (zero-cost fallback).")

    print(f"\n📂  Found {len(accounts_to_run)} account(s): {list(accounts_to_run.keys())}\n")

    for account_id, files in accounts_to_run.items():
        if args.phase in ("a", "both"):
            try:
                run_pipeline_a(account_id, files, use_ollama)
            except Exception as e:
                print(f"  [ERROR] Pipeline A failed for {account_id}: {e}")

        if args.phase in ("b", "both"):
            if not files.get("onboarding_file"):
                print(f"  [SKIP] Pipeline B skipped for {account_id} — no onboarding file.")
                continue
            try:
                run_pipeline_b(account_id, files, use_ollama)
            except Exception as e:
                print(f"  [ERROR] Pipeline B failed for {account_id}: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
