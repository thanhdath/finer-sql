#!/usr/bin/env python3
# This file was moved from /home/datht/mats/model_pool_generate.py
# Content intentionally identical to keep behavior unchanged.

from __future__ import annotations

import argparse
import json
import os
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv
try:
    import tiktoken  # optional, for OpenAI-style tokenization
except Exception:
    tiktoken = None
try:
    from transformers import AutoTokenizer  # optional, for HF tokenization
except Exception:
    AutoTokenizer = None
try:
    from datasets import load_from_disk, Dataset  # optional, for HF data loading
except Exception:
    load_from_disk = None
    Dataset = None


SYSTEM_PROMPT_FOR_REASONING_MODEL = '''
You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.
Follow this exact response format:

Rules:
- Output exactly one SQL statement.
- The SQL must be executable on SQLite.
- Do not include any explanatory text.
- Output one SQL statement only. Do not include any extra text, tags, or code fences.
'''

# SYSTEM_PROMPT_FOR_NON_REASONING_MODEL = '''
# You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.
# Follow this exact response format:

# Rules:
# - Output exactly one SQL statement.
# - The SQL must be executable on SQLite.
# - Do not include any explanatory text outside the <think> section.
# - After </think>, output one SQL statement only. Do not include any extra text, tags, or code fences.
# '''

SYSTEM_PROMPT_FOR_NON_REASONING_MODEL = '''
You are a meticulous SQL expert. Generate a single, correct SQL query for the user question and the provided database schema.
Follow this exact response format:

Rules:
- Output exactly one SQL statement.
- The SQL must be executable on SQLite.
- Think step by step and output the reasoning process in the <think> section.
- After </think>, output one SQL statement only. Do not include any extra text, tags, or code fences.
- Do not include any explanatory text outside the <think> section.
'''

@dataclass(frozen=True)
class ModelInfo:
    name: str
    provider: str  # openai | deepseek | together | google | local
    model_id: str
    reasoning: bool
    extra_params: Optional[Dict[str, Any]] = None


class ModelPool:
    def __init__(self, models: List[ModelInfo]):
        self.models = models

    def list_names(self) -> List[str]:
        return [m.name for m in self.models]

    def get(self, name: str) -> ModelInfo:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)


def default_model_pool() -> ModelPool:
    models = [
        # Two GPT-5 configs with reasoning_effort encoded in the model name
        # ModelInfo(name="gpt-5", provider="openai", model_id="gpt-5", reasoning=True, extra_params={"reasoning_effort": "low"}),
       ModelInfo(name="gpt-4.1-mini", provider="openai", model_id="gpt-4.1-mini", reasoning=False),
        # Local OpenAI-compatible server
        # ModelInfo(name="qwen2.5-72b", provider="local", model_id="qwen2.5-72b", reasoning=False),
        # deepseek
        # ModelInfo(name="deepseek-reasoner", provider="deepseek", model_id="deepseek-reasoner", reasoning=True),
        # OpenRouter reasoning models
        # ModelInfo(name="openai/gpt-oss-20b:free", provider="openrouter", model_id="openai/gpt-oss-20b:free", reasoning=True),
        ModelInfo(name="openai/gpt-oss-120b", provider="openrouter", model_id="openai/gpt-oss-120b", reasoning=True),
    ]
    return ModelPool(models)


class RateLimit(Exception):
    pass


def _exp_backoff_sleep(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    wait = min(cap, base * (2 ** attempt))
    jitter = random.uniform(0, 0.25 * wait)
    time.sleep(wait + jitter)


_openai_client: Optional[OpenAI] = None
_openai_local_client: Optional[OpenAI] = None
_openrouter_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def _get_openai_local_client() -> OpenAI:
    global _openai_local_client
    if _openai_local_client is None:
        # Local compatible OpenAI server
        base_url = os.getenv("OPENAI_LOCAL_BASE_URL", "http://100.88.196.45:8107/v1/")
        # Some local servers accept any string for api_key
        api_key = os.getenv("OPENAI_LOCAL_API_KEY", "sk-local")
        _openai_local_client = OpenAI(base_url=base_url, api_key=api_key)
    return _openai_local_client

def _get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            # default_headers={
            #     "HTTP-Referer": "https://github.com/your-repo",  # Optional: replace with your repo URL
            #     "X-Title": "SQL Writer"  # Optional: replace with your app name
            # }
        )
    return _openrouter_client

def call_openai_chat(model_id: str, messages: List[Dict[str, str]], *, temperature: Optional[float] = None, timeout: int = 60, extra_params: Optional[Dict[str, Any]] = {"reasoning_effort": "low"}) -> str:
    client = _get_openai_client()
    is_reasoning = False
    if 'gpt-5' in model_id:
        is_reasoning = True

    if is_reasoning:
        messages[0]["content"] = SYSTEM_PROMPT_FOR_REASONING_MODEL
    else:
        messages[0]["content"] = SYSTEM_PROMPT_FOR_NON_REASONING_MODEL

    kwargs: Dict[str, Any] = {
        "model": model_id, 
        "messages": messages,
    }
    if is_reasoning:
        kwargs["text"] = {"verbosity": "high"}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if extra_params:
        kwargs.update(extra_params)
    resp = client.chat.completions.create(**kwargs)
    if is_reasoning:
        reasoning_content = resp.choices[0].message.reasoning_content
        answer = resp.choices[0].message.content
        return_answer = "<think>\n" + reasoning_content + "\n</think>\n" + answer.strip()
    else:
        reasoning_content = None
        answer = resp.choices[0].message.content
        return_answer = answer.strip()
    print(return_answer)
    return return_answer.strip()


def call_deepseek_chat(model_id: str, messages: List[Dict[str, str]], *, temperature: float = 0.2, timeout: int = 60) -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    url = "https://api.deepseek.com/chat/completions"

    # replace system prompt for reasoning model
    messages[0]["content"] = SYSTEM_PROMPT_FOR_REASONING_MODEL

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 429:
        raise RateLimit(resp.text)
    if resp.status_code >= 400:
        raise RuntimeError(f"DeepSeek error {resp.status_code}: {resp.text}")
    data = resp.json()
    reasoning_content = data["choices"][0]["message"]["reasoning_content"]
    answer = data["choices"][0]["message"]["content"]
    return_answer = "<think>\n" + reasoning_content + "\n</think>\n" + answer.strip()
    print(return_answer)
    return return_answer.strip()


def call_together_chat(model_id: str, messages: List[Dict[str, str]], *, temperature: float = 0.2, timeout: int = 60) -> str:
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY missing")
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "temperature": temperature}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 429:
        raise RateLimit(resp.text)
    if resp.status_code >= 400:
        raise RuntimeError(f"Together error {resp.status_code}: {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_openrouter_chat(model_id: str, messages: List[Dict[str, str]], *, temperature: float = 0.2, timeout: int = 60, reasoning: bool = True) -> str:
    client = _get_openrouter_client()
    if reasoning:
        messages[0]["content"] = SYSTEM_PROMPT_FOR_REASONING_MODEL
    else:
        messages[0]["content"] = SYSTEM_PROMPT_FOR_NON_REASONING_MODEL

    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        timeout=timeout
    )
    reasoning_content = resp.choices[0].message.reasoning
    answer = resp.choices[0].message.content
    return_answer = "<think>\n" + reasoning_content + "\n</think>\n" + answer.strip()
    print(return_answer)
    return return_answer.strip()


def _gemini_from_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if not content:
            continue
        if role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})
    return contents


def call_gemini_generate(model_path: str, *, messages: Optional[List[Dict[str, str]]] = None, prompt_text: Optional[str] = None, temperature: float = 0.2, timeout: int = 60) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing")
    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent?key={api_key}"
    if messages:
        contents = _gemini_from_messages(messages)
    elif prompt_text is not None:
        contents = [{"role": "user", "parts": [{"text": prompt_text}]}]
    else:
        raise RuntimeError("Gemini call requires messages or prompt_text")
    payload = {"contents": contents, "generationConfig": {"temperature": temperature}}
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code == 429:
        raise RateLimit(resp.text)
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini error {resp.status_code}: {resp.text}")
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini empty response: {json.dumps(data)[:200]}")
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "\n".join([t for t in text_parts if t]).strip()


def generate_with_retries(model: ModelInfo, prompt: Optional[str], messages: Optional[List[Dict[str, str]]] = None, *, retry_max: int = 6, timeout: int = 60, gpt5_reasoning_effort: Optional[str] = None) -> str:
    attempt = 0
    while True:
        try:
            if model.provider == "google":
                return call_gemini_generate(model.model_id, messages=messages if messages else None, prompt_text=None if messages else (prompt or ""), timeout=timeout)
            chat_messages = messages if messages else [
                {"role": "user", "content": prompt or ""},
            ]
            if model.provider == "openai":
                # Use model.extra_params (e.g., reasoning_effort) baked into pool
                extra = model.extra_params or {}
                # Temperature default 0.2; some models only support default -> let API handle or set 1.0 if known
                restricted = {"o4-mini", "gpt-5-mini"}
                temp = None if model.model_id.startswith("gpt-5") else (1.0 if model.model_id in restricted else 0.2)
                return call_openai_chat(model.model_id, chat_messages, temperature=temp, timeout=timeout, extra_params=extra)
            if model.provider == "local":
                # Use OpenAI-compatible local server
                client = _get_openai_local_client()
                try:
                    resp = client.chat.completions.create(model=model.model_id, messages=chat_messages)
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    raise RuntimeError(f"Local OpenAI error: {str(e)}")
            if model.provider == "deepseek":
                return call_deepseek_chat(model.model_id, chat_messages, timeout=timeout)
            if model.provider == "together":
                return call_together_chat(model.model_id, chat_messages, timeout=timeout)
            if model.provider == "openrouter":
                return call_openrouter_chat(model.model_id, chat_messages, timeout=timeout, reasoning=model.reasoning)
            raise RuntimeError(f"Unknown provider: {model.provider}")
        except RateLimit as e:
            if attempt >= retry_max:
                raise RuntimeError(f"RateLimit after {attempt} retries: {e}")
            base = 4.0 if model.provider == "together" else 1.0
            _exp_backoff_sleep(attempt, base=base, cap=90.0)
            attempt += 1
        except requests.RequestException as e:
            if attempt >= retry_max:
                raise RuntimeError(f"HTTP error after {attempt} retries: {e}")
            _exp_backoff_sleep(attempt, base=2.0, cap=60.0)
            attempt += 1


def ensure_indexes(coll) -> None:
    try:
        coll.create_index([("model_name", ASCENDING), ("sample_id", ASCENDING)], unique=True, background=True)
    except Exception:
        pass


def doc_already_processed(out_coll, model_name: str, sample_id: Any) -> bool:
    return out_coll.find_one({"model_name": model_name, "sample_id": sample_id}, {"_id": 1}) is not None


def process_one_sample(sample: Dict[str, Any], pool: ModelPool, args) -> Tuple[Any, Dict[str, Any]]:
    sample_id = sample.get(args.id_field)
    prompt = sample.get(args.prompt_field)
    messages = sample.get("messages")
    if not prompt and not messages:
        return sample_id, {"status": "skipped", "error": "missing prompt"}

    results: Dict[str, Any] = {"status": "ok", "count": 0, "errors": {}}
    for model in pool.models:
        if args.models and model.name not in args.models:
            continue
        try:
            resp_text = generate_with_retries(model, prompt if not messages else None, messages=messages if messages else None, retry_max=args.retry_max, timeout=args.timeout)
            results[model.name] = {"ok": True, "response": resp_text}
            results["count"] += 1
        except Exception as e:
            results[model.name] = {"ok": False, "error": str(e)}
            results["errors"][model.name] = str(e)
    return sample_id, results


def worker_process(doc: Dict[str, Any], args, needed_models: List[str], skip_map: Dict[Any, set]) -> Tuple[Any, Dict[str, Any]]:
    """Top-level worker: generate and write per-model immediately to MongoDB."""
    local_pool = default_model_pool()
    sid = doc.get(args.id_field)

    # Always use dataset-provided messages
    messages_input = doc.get("messages") or []
    base_prompt = None

    # Mongo per-process client
    client = MongoClient(args.mongo_uri)
    out_coll = client["mats"][args.output_collection]

    written = 0
    skipped = 0
    errors: Dict[str, str] = {}

    # Determine models to run, honoring retry_failed to select only missing/failed
    model_names = needed_models or local_pool.list_names()
    name_to_info = {m.name: m for m in local_pool.models}

    # Skip entire sample if fully processed
    if args.skip_processed and not args.overwrite and not args.retry_failed:
        already = skip_map.get(sid, set())
        needed_set = set(model_names)
        if needed_set.issubset(already):
            return sid, {"status": "skipped_all"}

    for name in model_names:
        model = name_to_info.get(name)
        if model is None:
            continue
        # Skip/Retry logic
        if args.skip_processed and not args.overwrite and not args.retry_failed:
            if out_coll.find_one({"model_name": name, "sample_id": sid}, {"_id": 1}):
                skipped += 1
                continue
        if args.retry_failed:
            existing = out_coll.find_one({"model_name": name, "sample_id": sid}, {"ok": 1, "response": 1})
            if existing and (existing.get("ok") is True and existing.get("response")):
                skipped += 1
                continue
        # Generate
        ok = True
        response_text: Optional[str] = None
        err_msg: Optional[str] = None
        try:
            response_text = generate_with_retries(
                model,
                None,
                messages=messages_input,
                retry_max=args.retry_max,
                timeout=args.timeout,
                gpt5_reasoning_effort=getattr(args, 'gpt5_reasoning_effort', None),
            )
        except Exception as e:
            ok = False
            err_msg = str(e)
            errors[name] = err_msg
        # Build doc and write immediately
        # Adjust model_name with reasoning effort when applicable
        eff = (model.extra_params or {}).get("reasoning_effort")
        eff_suffix = f"--{eff}" if eff in {"low","medium","high"} else ""
        stored_model_name = f"{name}{eff_suffix}"
        out_doc = {
            "model_name": stored_model_name,
            "provider": model.provider,
            "reasoning": model.reasoning,
            "sample_id": sid,
            "prompt": None,
            "response": response_text,
            "ok": ok,
            "error": None if ok else err_msg,
            "created_at": datetime.utcnow(),
        }
        out_doc["messages"] = messages_input
        # Include original metadata if present
        meta = doc.get("meta") or {}
        if meta:
            out_doc.update(meta)
        filt = {"model_name": stored_model_name, "sample_id": sid}
        if args.overwrite:
            out_coll.replace_one(filt, out_doc, upsert=True)
            written += 1
        else:
            try:
                out_coll.insert_one({**filt, **out_doc})
                written += 1
            except DuplicateKeyError:
                skipped += 1

    status = {"status": "ok", "written": written, "skipped": skipped, "errors": errors}
    return sid, status


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("LLM model pool generator (HF bird only) → MongoDB output")
    ap.add_argument("--mongo_uri", default="mongodb://localhost:27017")
    ap.add_argument("--output_collection", default="llm_pool_bird", help="mats.<collection> to store results")
    ap.add_argument("--id_field", default="sample_id")
    ap.add_argument("--prompt_field", default="prompt_raw")
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--limit", type=int, default=10, help="0 = all; default 10 for safe runs")
    ap.add_argument("--skip_processed", default=True, action="store_true", help="skip if model_name+sample_id exists")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--models", nargs="*", help="subset of model names to run")
    ap.add_argument("--gpt5_reasoning_effort", choices=["low","medium","high"], help="Reasoning effort for GPT-5; stored in model_name suffix")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retry_max", type=int, default=6)
    # Input: local HF datasets only (bird)
    ap.add_argument("--data_root", default="/home/datht/graph-schema/end2end/data", help="Root dir of saved HF datasets (bird only)")
    # Retry/repair options
    ap.add_argument("--retry_failed", action="store_true", help="Only re-run models with missing/failed responses in output collection")
    # confirmation + tokenization options
    ap.add_argument("--confirm", action="store_true", help="Compute total tokens and require 'yes' before generating")
    ap.add_argument("--tok_mode", choices=["openai","hf"], default="openai", help="Token counter: openai (tiktoken) or hf (transformers)")
    ap.add_argument("--tok_model", default="cl100k_base", help="tiktoken encoding (e.g., cl100k_base, o200k_base)")
    ap.add_argument("--tok_hf_path", default="/home/datht/huggingface/Qwen/Qwen2.5-0.5B-Instruct", help="HF tokenizer path if tok_mode=hf")
    ap.add_argument("--verbose", action="store_true", help="Print debug information while loading data")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load environment variables from /home/datht/mats/.env if present
    try:
        load_dotenv("/home/datht/mats/.env")
    except Exception:
        pass

    pool = default_model_pool()
    if args.models:
        unknown = [m for m in args.models if m not in pool.list_names()]
        if unknown:
            raise SystemExit(f"Unknown models: {unknown}. Available: {pool.list_names()}")

    client = MongoClient(args.mongo_uri)
    out_coll = client["mats"][args.output_collection]
    ensure_indexes(out_coll)

    # Build samples from HF datasets (bird only)
    samples: List[Dict[str, Any]] = []
    samples_by_id: Dict[Any, Dict[str, Any]] = {}
    if load_from_disk is None:
        raise SystemExit("datasets library not available; please install datasets")
    if not os.path.isdir(args.data_root):
        raise SystemExit(f"Data root not found: {args.data_root}")
    combined_rows: List[Dict[str, Any]] = []
    if args.verbose:
        print(f"[load] Scanning data_root: {args.data_root}")
    for name in sorted(os.listdir(args.data_root)):
        path = os.path.join(args.data_root, name)
        if not os.path.isdir(path):
            continue
        try:
            ds_or_dd = load_from_disk(path)
        except Exception as e1:
            # Try common split subdir fallback
            try:
                ds_or_dd = load_from_disk(os.path.join(path, "train"))
            except Exception as e2:
                if args.verbose:
                    print(f"[load] Skipped (load error): {path} | {type(e1).__name__}: {e1} | fallback: {type(e2).__name__}: {e2}")
                continue
        if hasattr(ds_or_dd, "keys") and callable(getattr(ds_or_dd, "keys", None)):
            # Prefer train, else first available split
            if "train" in ds_or_dd:
                part = ds_or_dd["train"]
            else:
                keys = list(ds_or_dd.keys())
                if not keys:
                    if args.verbose:
                        print(f"[load] Skipped (empty DatasetDict): {path}")
                    continue
                part = ds_or_dd[keys[0]]
        else:
            part = ds_or_dd
        combined_rows.extend(part.to_list())
    if args.verbose:
        print(f"[load] Loaded rows total: {len(combined_rows)}")
    # Filter BIRD only
    # combined_rows = [r for r in combined_rows if (r.get("dataset_name") == "bird" or r.get("dataset") == "bird")]
    if args.verbose:
        print(f"[load] Rows after bird filter: {len(combined_rows)}")
    # Normalize into samples with messages and metadata pass-through
    for idx, r in enumerate(combined_rows):
        sid = r.get("sample_id") or r.get("_id") or r.get("question_id") or idx
        messages = r.get("messages")
        if messages is None:
            if args.verbose and idx < 5:
                print(f"[load] Skipping row without messages. sid={sid}")
            continue
        # Select important metadata to store with outputs
        meta: Dict[str, Any] = {}
        # dataset name
        if r.get("dataset_name") is not None:
            meta["dataset_name"] = r.get("dataset_name")
        elif r.get("dataset") is not None:
            meta["dataset_name"] = r.get("dataset")
        # common fields
        for k in [
            "split",
            "db_id",
            "question",
            "groundtruth_sqls",
            "SQL",
            "sql",
        ]:
            if r.get(k) is not None:
                meta[k] = r.get(k)

        meta["ground_truth_sql"] = meta["groundtruth_sqls"][0]
        doc = {args.id_field: sid, "messages": messages, "meta": meta}
        samples.append(doc)
        samples_by_id[sid] = doc
        if args.limit and len(samples) >= args.limit:
            break
    if args.verbose:
        print(f"[load] Prepared samples: {len(samples)} (limit={args.limit})")

    if not samples:
        print("No samples to process.")
        return

    # Optional: compute token stats and ask for confirmation
    if args.confirm:
        def to_prompt_text(doc: Dict[str, Any]) -> str:
            prompt = doc.get(args.prompt_field)
            messages = doc.get("messages")
            if prompt:
                return str(prompt)
            if messages:
                # concatenate user+system+assistant for counting; generation uses user-only when prompt missing
                parts = []
                for m in messages:
                    c = m.get("content", "")
                    if c:
                        parts.append(str(c))
                return "\n\n".join(parts)
            return ""

        texts = [to_prompt_text(d) for d in samples]

        def count_tokens_batch(texts: List[str]) -> Tuple[int, List[int]]:
            counts: List[int] = []
            if args.tok_mode == "openai" and tiktoken is not None:
                try:
                    enc = tiktoken.get_encoding(args.tok_model)
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base") if tiktoken is not None else None
                if enc is not None:
                    for t in texts:
                        counts.append(len(enc.encode(t or "")))
                    return sum(counts), counts
            if args.tok_mode == "hf" and AutoTokenizer is not None and args.tok_hf_path:
                try:
                    tok = AutoTokenizer.from_pretrained(args.tok_hf_path, use_fast=True)
                    for t in texts:
                        counts.append(len(tok.encode(t or "", add_special_tokens=False)))
                    return sum(counts), counts
                except Exception:
                    pass
            # fallback: whitespace tokens
            for t in texts:
                counts.append(len((t or "").split()))
            return sum(counts), counts

        total_tokens, per_sample = count_tokens_batch(texts)
        avg = (total_tokens / max(1, len(per_sample)))
        print(f"Planned generation preview: {len(samples)} samples")
        print(f"Tokenization mode: {args.tok_mode}{' ('+args.tok_model+')' if args.tok_mode=='openai' else (' ('+args.tok_hf_path+')' if args.tok_mode=='hf' and args.tok_hf_path else '')}")
        print(f"Total tokens: {total_tokens} | Avg per sample: {avg:.1f}")
        try:
            answer = input("Type 'yes' to proceed with generation, anything else to abort: ").strip().lower()
        except EOFError:
            answer = ""
        if answer != "yes":
            print("Aborted by user before calling models.")
            return

    skip_map: Dict[Any, set] = {}
    if args.skip_processed and not args.overwrite:
        for model_name in (args.models or pool.list_names()):
            existing = out_coll.find({"model_name": model_name}, {"sample_id": 1})
            done_ids = {d.get("sample_id") for d in existing}
            for sid in done_ids:
                skip_map.setdefault(sid, set()).add(model_name)

    from multiprocessing import Pool as ProcPool

    def _needed_models_list() -> List[str]:
        return args.models or pool.list_names()

    print(f"Processing {len(samples)} samples with up to {args.processes} processes and models: {_needed_models_list()}")

    # Prepare inputs for starmap to avoid pickling local closures
    worker_inputs = [(doc, args, _needed_models_list(), skip_map) for doc in samples]

    if args.processes <= 1:
        results = [worker_process(*wi) for wi in worker_inputs]
    else:
        with ProcPool(processes=args.processes) as p:
            results = p.starmap(worker_process, worker_inputs, chunksize=1)

    total_written = sum(res.get("written", 0) for _, res in results)
    total_skipped = sum(res.get("skipped", 0) for _, res in results)
    print(f"Done. Wrote {total_written} documents to mats.{args.output_collection} (skipped {total_skipped}).")


if __name__ == "__main__":
    main()


