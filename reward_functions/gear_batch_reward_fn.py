from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


load_dotenv()

MAX_CONCURRENT_WORKERS = int(os.getenv("GEAR_MAX_CONCURRENT_WORKERS", "16"))
GEAR_DEBUG_PRINT_EVERY = int(os.getenv("GEAR_DEBUG_PRINT_EVERY", "20"))
GEAR_DEBUG_PRINT_FIRST = int(os.getenv("GEAR_DEBUG_PRINT_FIRST", "3"))
GEAR_GROUP_RESPONSES_PER_PROMPT = os.getenv("GEAR_GROUP_RESPONSES_PER_PROMPT", "1").lower() not in {
    "0",
    "false",
    "no",
}
GEAR_MAX_RESPONSES_PER_JUDGE = int(os.getenv("GEAR_MAX_RESPONSES_PER_JUDGE", "4"))

GEAR_DEBUG_JUDGE = os.getenv("GEAR_DEBUG_JUDGE", "0").lower() in {"1", "true", "yes"}
GEAR_FAIL_ON_PARSE_ERROR = os.getenv("GEAR_FAIL_ON_PARSE_ERROR", "0").lower() in {"1", "true", "yes"}
GEAR_DEBUG_DIR = os.getenv("GEAR_DEBUG_DIR", "./log/reward_debug")
GEAR_DEBUG_MAX_FILES = int(os.getenv("GEAR_DEBUG_MAX_FILES", "200"))
GEAR_DEBUG_RAW_RESPONSE_CHARS = int(os.getenv("GEAR_DEBUG_RAW_RESPONSE_CHARS", "12000"))
GEAR_DEBUG_ROLLOUT_CHARS = int(os.getenv("GEAR_DEBUG_ROLLOUT_CHARS", "2000"))


_GEAR_SPEC = importlib_util.spec_from_file_location(
    "gear_reward_score",
    Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "gear.py",
)
_GEAR_MODULE = importlib_util.module_from_spec(_GEAR_SPEC)
assert _GEAR_SPEC.loader is not None
sys.modules[_GEAR_SPEC.name] = _GEAR_MODULE
_GEAR_SPEC.loader.exec_module(_GEAR_MODULE)
aggregate_gear_reward = _GEAR_MODULE.aggregate_gear_reward

_RULE_FN_SPEC = importlib_util.spec_from_file_location(
    "rule_fn_module",
    Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / "rule_fn.py",
)
_RULE_FN_MODULE = importlib_util.module_from_spec(_RULE_FN_SPEC)
assert _RULE_FN_SPEC.loader is not None
sys.modules[_RULE_FN_SPEC.name] = _RULE_FN_MODULE
_RULE_FN_SPEC.loader.exec_module(_RULE_FN_MODULE)
get_verification_function = _RULE_FN_MODULE.get_verification_function

_reward_call_count = 0
_reward_call_lock = threading.Lock()

_gear_debug_file_count = 0
_gear_debug_file_lock = threading.Lock()


JUDGE_TEMPLATE_BATCH_BINARY = """
Your job is to evaluate the last assistant response in a conversation against multiple rubric items.

# Conversation
<<conversation>>

# Rubric Items
<<rubric_items>>

# Instructions
Return exactly one JSON object and nothing else.

Required rubric keys:
<<required_keys>>

The JSON object is invalid if any required key is missing.
The JSON object is invalid if it contains fewer than <<rubric_count>> rubric judgments.
Do not stop before returning all required keys.
Do not use rubric ids such as "r1".
Do not include explanations.
Do not include any text outside the JSON object.

For each rubric item, return an object with this exact schema:
{
  "criteria_met": true or false
}

Meaning:
- criteria_met=true means the assistant response satisfies this rubric item.
- criteria_met=false means the assistant response does not satisfy this rubric item.
- For negative-point rubrics, criteria_met=true means the undesirable criterion is present.
- For negative-point rubrics, if the undesirable criterion is not present, return criteria_met=false and prob_met close to 0.0.

Example:
{
  "1": {"criteria_met": true},
  "2": {"criteria_met": false}
}
""".strip()


JUDGE_TEMPLATE_BATCH_PROB = """
Your job is to evaluate the last assistant response in a conversation against multiple rubric items.

# Conversation
<<conversation>>

# Rubric Items
<<rubric_items>>

# Instructions
Return exactly one JSON object and nothing else.

Required rubric keys:
<<required_keys>>

The JSON object is invalid if any required key is missing.
The JSON object is invalid if it contains fewer than <<rubric_count>> rubric judgments.
Do not stop before returning all required keys.
Do not use rubric ids such as "r1".
Do not use labels such as "PRESENT" or "NOT_PRESENT".
Do not include explanations.
Do not include any text outside the JSON object.

For each rubric item, return an object with this exact schema:
{
  "criteria_met": true or false,
  "prob_met": a number between 0 and 1
}

Meaning:
- criteria_met=true means the assistant response satisfies this rubric item.
- criteria_met=false means the assistant response does not satisfy this rubric item.
- prob_met means the probability that criteria_met is true.
- If criteria_met is true, prob_met should be >= 0.5.
- If criteria_met is false, prob_met should be <= 0.5.
- For negative-point rubrics, criteria_met=true means the undesirable criterion is present.
- For negative-point rubrics, if the undesirable criterion is not present, return criteria_met=false and prob_met close to 0.0.

Example:
{
  "1": {"criteria_met": true, "prob_met": 0.95},
  "2": {"criteria_met": false, "prob_met": 0.10}
}
""".strip()


JUDGE_TEMPLATE_MULTI_RESPONSE_BINARY = """
Your job is to evaluate multiple candidate assistant responses to the same conversation against multiple rubric items.

# Conversation Before Candidate Responses
<<conversation>>

# Candidate Assistant Responses
<<responses>>

# Rubric Items
<<rubric_items>>

# Instructions
Return exactly one JSON object and nothing else.

Required top-level response keys:
<<required_response_keys>>

For each response, required rubric keys:
<<required_rubric_keys>>

The JSON object is invalid if any required response key is missing.
Each response object is invalid if any required rubric key is missing.
Do not stop before returning all required response keys and all required rubric keys.
Do not use rubric ids such as "r1".
Do not include explanations.
Do not include any text outside the JSON object.

For each response and rubric item, return an object with this exact schema:
{
  "criteria_met": true or false
}

Meaning:
- criteria_met=true means the candidate response satisfies this rubric item.
- criteria_met=false means the candidate response does not satisfy this rubric item.
- For negative-point rubrics, criteria_met=true means the undesirable criterion is present.
- For negative-point rubrics, if the undesirable criterion is not present, return criteria_met=false and prob_met close to 0.0.

Example:
{
  "1": {
    "1": {"criteria_met": true},
    "2": {"criteria_met": false}
  },
  "2": {
    "1": {"criteria_met": false},
    "2": {"criteria_met": true}
  }
}
""".strip()


JUDGE_TEMPLATE_MULTI_RESPONSE_PROB = """
Your job is to evaluate multiple candidate assistant responses to the same conversation against multiple rubric items.

# Conversation Before Candidate Responses
<<conversation>>

# Candidate Assistant Responses
<<responses>>

# Rubric Items
<<rubric_items>>

# Instructions
Return exactly one JSON object and nothing else.

Required top-level response keys:
<<required_response_keys>>

For each response, required rubric keys:
<<required_rubric_keys>>

The JSON object is invalid if any required response key is missing.
Each response object is invalid if any required rubric key is missing.
Do not stop before returning all required response keys and all required rubric keys.
Do not use rubric ids such as "r1".
Do not use labels such as "PRESENT" or "NOT_PRESENT".
Do not include explanations.
Do not include any text outside the JSON object.

For each response and rubric item, return an object with this exact schema:
{
  "criteria_met": true or false,
  "prob_met": a number between 0 and 1
}

Meaning:
- criteria_met=true means the candidate response satisfies this rubric item.
- criteria_met=false means the candidate response does not satisfy this rubric item.
- prob_met means the probability that criteria_met is true.
- If criteria_met is true, prob_met should be >= 0.5.
- If criteria_met is false, prob_met should be <= 0.5.
- For negative-point rubrics, criteria_met=true means the undesirable criterion is present.
- For negative-point rubrics, if the undesirable criterion is not present, return criteria_met=false and prob_met close to 0.0.

Example:
{
  "1": {
    "1": {"criteria_met": true, "prob_met": 0.95},
    "2": {"criteria_met": false, "prob_met": 0.10}
  },
  "2": {
    "1": {"criteria_met": false, "prob_met": 0.25},
    "2": {"criteria_met": true, "prob_met": 0.80}
  }
}
""".strip()


@dataclass
class RubricItem:
    id: str
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, rubric: Dict[str, Any], rubric_idx: int) -> "RubricItem":
        tags_data = rubric.get("tags", {})
        if isinstance(tags_data, list):
            parsed_tags = {}
            for tag in tags_data:
                if isinstance(tag, str) and ":" in tag:
                    key, value = tag.split(":", 1)
                    parsed_tags[key] = value
                elif isinstance(tag, str):
                    parsed_tags[tag] = True
            tags_data = parsed_tags
        elif not isinstance(tags_data, dict):
            tags_data = {}

        return cls(
            id=rubric.get("id", f"r{rubric_idx + 1}"),
            criterion=rubric["criterion"],
            points=float(rubric["points"]),
            tags=tags_data,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags,
        }


@dataclass
class SamplerResponse:
    response_text: str
    response_metadata: Dict[str, Any]
    actual_queried_message_list: List[Dict[str, str]]


@dataclass
class BatchJudgeParseResult:
    results: Dict[int, Dict[str, Any]]
    parse_status: str
    missing_count: int


def _debug_enabled() -> bool:
    return GEAR_DEBUG_JUDGE


def _fail_on_parse_error() -> bool:
    return GEAR_FAIL_ON_PARSE_ERROR


def _safe_name(value: Any, max_len: int = 120) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:max_len] or "unknown"


def _debug_env_snapshot() -> Dict[str, Any]:
    return {
        "VLLM_BASE_URL": os.getenv("VLLM_BASE_URL"),
        "VLLM_MODEL": os.getenv("VLLM_MODEL"),
        "VLLM_MAX_TOKENS": os.getenv("VLLM_MAX_TOKENS"),
        "VLLM_TIMEOUT": os.getenv("VLLM_TIMEOUT"),
        "VLLM_LOAD_REFRESH_INTERVAL_SEC": os.getenv("VLLM_LOAD_REFRESH_INTERVAL_SEC"),
        "GEAR_MAX_CONCURRENT_WORKERS": os.getenv("GEAR_MAX_CONCURRENT_WORKERS"),
        "GEAR_GROUP_RESPONSES_PER_PROMPT": os.getenv("GEAR_GROUP_RESPONSES_PER_PROMPT"),
        "GEAR_MAX_RESPONSES_PER_JUDGE": os.getenv("GEAR_MAX_RESPONSES_PER_JUDGE"),
        "GEAR_DEBUG_JUDGE": os.getenv("GEAR_DEBUG_JUDGE"),
        "GEAR_FAIL_ON_PARSE_ERROR": os.getenv("GEAR_FAIL_ON_PARSE_ERROR"),
        "GEAR_DEBUG_DIR": os.getenv("GEAR_DEBUG_DIR"),
    }


def _write_debug_file(
    *,
    tag: str,
    sample_key: Any,
    attempt_idx: int,
    grader_prompt: str,
    raw_response: str,
    parse_status: Optional[str],
    missing_count: Optional[int],
    expected_count: Optional[int],
    pi_mode: str,
    response_metadata: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
    extra: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> Dict[str, str]:
    global _gear_debug_file_count

    if not force and not _debug_enabled():
        return {}

    with _gear_debug_file_lock:
        if _gear_debug_file_count >= GEAR_DEBUG_MAX_FILES:
            return {}
        _gear_debug_file_count += 1
        file_idx = _gear_debug_file_count

    debug_dir = Path(GEAR_DEBUG_DIR).resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

    safe_sample = _safe_name(sample_key)
    safe_tag = _safe_name(tag)
    base_name = f"{file_idx:05d}_{safe_tag}_sample_{safe_sample}_attempt_{attempt_idx}_pid_{os.getpid()}"

    json_path = debug_dir / f"{base_name}.json"
    txt_path = debug_dir / f"{base_name}.txt"

    raw_limit = max(0, GEAR_DEBUG_RAW_RESPONSE_CHARS)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pid": os.getpid(),
        "tag": tag,
        "sample_key": sample_key,
        "attempt_idx": attempt_idx,
        "expected_count": expected_count,
        "missing_count": missing_count,
        "pi_mode": pi_mode,
        "parse_status": parse_status,
        "response_metadata": response_metadata or {},
        "exception": repr(exception) if exception is not None else None,
        "traceback": traceback.format_exc() if exception is not None else None,
        "env": _debug_env_snapshot(),
        "raw_response_len": len(raw_response or ""),
        "raw_response_preview": (raw_response or "")[:raw_limit],
        "grader_prompt_len": len(grader_prompt or ""),
        "grader_prompt_preview": (grader_prompt or "")[:raw_limit],
        "extra": extra or {},
    }

    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp_json.replace(json_path)

    txt_path.write_text(
        "===== ENV =====\n"
        + json.dumps(payload["env"], ensure_ascii=False, indent=2, default=str)
        + "\n\n===== META =====\n"
        + json.dumps(
            {
                "timestamp": payload["timestamp"],
                "pid": payload["pid"],
                "tag": tag,
                "sample_key": sample_key,
                "attempt_idx": attempt_idx,
                "expected_count": expected_count,
                "missing_count": missing_count,
                "pi_mode": pi_mode,
                "parse_status": parse_status,
                "response_metadata": response_metadata or {},
                "exception": payload["exception"],
                "debug_json": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n\n===== GRADER PROMPT =====\n"
        + (grader_prompt or "")
        + "\n\n===== RAW GRADER RESPONSE =====\n"
        + (raw_response or "")
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[GEAR_DEBUG_JUDGE] wrote json={json_path} txt={txt_path} "
        f"parse_status={parse_status} raw_len={len(raw_response or '')} "
        f"VLLM_BASE_URL={os.getenv('VLLM_BASE_URL')}",
        flush=True,
    )

    return {"debug_json": str(json_path), "debug_txt": str(txt_path)}


def _rollout_debug_fields(
    *,
    raw_response: str,
    grader_prompt: str,
    response_metadata: Optional[Dict[str, Any]],
    debug_paths: Optional[Dict[str, str]],
    exception: Optional[str] = None,
) -> Dict[str, Any]:
    if not _debug_enabled():
        return {}

    limit = max(0, GEAR_DEBUG_ROLLOUT_CHARS)
    return {
        "llm_raw_response_preview": (raw_response or "")[:limit],
        "llm_grader_prompt_preview": (grader_prompt or "")[:limit],
        "llm_response_metadata": response_metadata or {},
        "llm_debug_json": (debug_paths or {}).get("debug_json", ""),
        "llm_debug_txt": (debug_paths or {}).get("debug_txt", ""),
        "llm_exception": exception or "",
        "llm_vllm_base_url": os.getenv("VLLM_BASE_URL", ""),
    }


class VLLMSampler:
    def __init__(
        self,
        base_urls: Optional[List[str]] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 300,
        load_refresh_interval_sec: float = 2.0,
    ):
        urls = base_urls
        if urls is None:
            url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
            urls = [url.strip() for url in url_env.split(",") if url.strip()]
        self.base_urls = urls or ["http://localhost:8001/v1"]
        self.model = model or os.getenv("VLLM_MODEL", "default")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = int(os.getenv("VLLM_TIMEOUT", str(timeout)))
        self.load_refresh_interval_sec = float(
            os.getenv("VLLM_LOAD_REFRESH_INTERVAL_SEC", str(load_refresh_interval_sec))
        )
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        }
        self._lock = threading.Lock()
        self._url_loads = {
            url: {"running": 0, "waiting": 0, "total": 0, "available": True}
            for url in self.base_urls
        }
        self._virtual_loads = {url: 0 for url in self.base_urls}
        self._last_refresh_ts = 0.0
        self._refresh_loads(force=True)

    def _metrics_url(self, base_url: str) -> str:
        normalized = base_url[:-3] if base_url.endswith("/v1") else base_url.rstrip("/")
        return f"{normalized}/metrics"

    def _parse_metric_value(self, metrics_text: str, metric_name: str) -> int:
        pattern = rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.]+)"
        matches = re.findall(pattern, metrics_text, flags=re.MULTILINE)
        if not matches:
            return 0
        return int(float(matches[0]))

    def _get_url_load(self, base_url: str) -> Dict[str, Any]:
        try:
            response = requests.get(self._metrics_url(base_url), timeout=5)
            if response.status_code != 200:
                return {"running": 0, "waiting": 0, "total": 0, "available": True}
            metrics_text = response.text
            running = self._parse_metric_value(metrics_text, "vllm:num_requests_running")
            waiting = self._parse_metric_value(metrics_text, "vllm:num_requests_waiting")
            return {
                "running": running,
                "waiting": waiting,
                "total": running + waiting,
                "available": True,
            }
        except Exception:
            return {"running": 0, "waiting": 0, "total": 0, "available": True}

    def _refresh_loads(self, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if not force and now - self._last_refresh_ts < self.load_refresh_interval_sec:
                return
            current_urls = list(self.base_urls)
        fresh_loads = {url: self._get_url_load(url) for url in current_urls}
        with self._lock:
            for url, load_info in fresh_loads.items():
                self._url_loads[url] = load_info
            self._last_refresh_ts = now

    def _acquire_url(self) -> str:
        self._refresh_loads()
        with self._lock:
            available_urls = [url for url in self.base_urls if self._url_loads[url].get("available", True)]
            if not available_urls:
                available_urls = list(self.base_urls)
            selected = min(
                available_urls,
                key=lambda url: self._url_loads[url]["total"] + self._virtual_loads[url],
            )
            self._virtual_loads[selected] += 1
            return selected

    def _release_url(self, base_url: str) -> None:
        with self._lock:
            self._virtual_loads[base_url] = max(0, self._virtual_loads[base_url] - 1)

    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        payload = {
            "model": self.model,
            "messages": message_list,
            "temperature": self.temperature,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        }

        last_error = None
        last_url = None
        for trial in range(5):
            url = self._acquire_url()
            last_url = url
            try:
                response = requests.post(
                    f"{url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                if not content:
                    raise ValueError("Empty content returned from VLLM judge")
                return SamplerResponse(
                    response_text=content.strip(),
                    response_metadata={
                        "usage": response_data.get("usage", {}),
                        "base_url": url,
                        "model": self.model,
                        "status_code": response.status_code,
                        "trial": trial,
                    },
                    actual_queried_message_list=message_list,
                )
            except Exception as exc:
                last_error = exc
                if _debug_enabled():
                    print(
                        f"[GEAR][WARN] VLLM judge request failed trial={trial} "
                        f"url={url} error={repr(exc)}",
                        flush=True,
                    )
                time.sleep(min(2**trial, 8))
            finally:
                self._release_url(url)

        raise RuntimeError(f"Failed to query VLLM judge after retries. last_url={last_url}, last_error={last_error}") from last_error


_global_grader: Optional[VLLMSampler] = None


def get_global_grader() -> VLLMSampler:
    global _global_grader
    if _global_grader is None:
        _global_grader = VLLMSampler(max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "2048")))
        print(
            f"[GEAR] grader urls={_global_grader.base_urls}, "
            f"model={_global_grader.model}, timeout={_global_grader.timeout}, "
            f"load_refresh_interval_sec={_global_grader.load_refresh_interval_sec}, "
            f"max_workers_per_url={MAX_CONCURRENT_WORKERS}, "
            f"group_responses_per_prompt={GEAR_GROUP_RESPONSES_PER_PROMPT}, "
            f"max_responses_per_judge={GEAR_MAX_RESPONSES_PER_JUDGE}, "
            f"max_tokens={_global_grader.max_tokens}, "
            f"debug_judge={GEAR_DEBUG_JUDGE}, "
            f"fail_on_parse_error={GEAR_FAIL_ON_PARSE_ERROR}, "
            f"debug_dir={GEAR_DEBUG_DIR}",
            flush=True,
        )
    return _global_grader


def parse_json_to_dict(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    parsed: Dict[str, Any] = {}
    criteria_match = re.search(r'"criteria_met"\s*:\s*(true|false)', stripped, flags=re.IGNORECASE)
    if criteria_match:
        parsed["criteria_met"] = criteria_match.group(1).lower() == "true"

    prob_match = re.search(r'"prob_met"\s*:\s*([0-9]*\.?[0-9]+)', stripped)
    if prob_match:
        parsed["prob_met"] = float(prob_match.group(1))

    return parsed


def _extract_json_candidates(text: str) -> List[str]:
    stripped = text.strip()
    candidates: List[str] = []
    if stripped:
        candidates.append(stripped)

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        block = block.strip()
        if block:
            candidates.append(block)

    first_brace = stripped.find("{")
    if first_brace != -1:
        brace_depth = 0
        for idx in range(first_brace, len(stripped)):
            char = stripped[idx]
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    candidate = stripped[first_brace : idx + 1].strip()
                    if candidate:
                        candidates.append(candidate)
                    break

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _parse_numeric_key(key: Any, max_count: int) -> Optional[int]:
    text = str(key).strip()
    match = re.fullmatch(r"(?:r|rubric|item|criterion)?[_\-\s]*(\d+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    idx = int(match.group(1))
    if 1 <= idx <= max_count:
        return idx
    return None


def _find_numeric_key_dict(payload: Any, expected_count: int) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    direct_numeric_keys = 0
    for key in payload:
        if _parse_numeric_key(key, expected_count) is not None:
            direct_numeric_keys += 1

    if direct_numeric_keys > 0:
        return payload

    best_nested: Optional[Dict[str, Any]] = None
    best_score = 0
    for value in payload.values():
        nested = _find_numeric_key_dict(value, expected_count)
        if nested is None:
            continue

        score = 0
        for key in nested:
            if _parse_numeric_key(key, expected_count) is not None:
                score += 1

        if score > best_score:
            best_score = score
            best_nested = nested

    return best_nested


def _numeric_key_count(payload: Any, expected_count: int) -> int:
    if not isinstance(payload, dict):
        return 0
    count = 0
    for key in payload:
        if _parse_numeric_key(key, expected_count) is not None:
            count += 1
    return count


def _find_multi_response_dict(payload: Any, response_count: int, rubric_count: int) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    response_key_count = _numeric_key_count(payload, response_count)
    response_score = 0
    if response_key_count > 0:
        for response_idx in range(1, response_count + 1):
            value = payload.get(str(response_idx), payload.get(response_idx))
            response_score += _numeric_key_count(value, rubric_count)
        if response_score > 0:
            return payload

    best_nested: Optional[Dict[str, Any]] = None
    best_score = 0
    for value in payload.values():
        nested = _find_multi_response_dict(value, response_count, rubric_count)
        if nested is None:
            continue

        score = 0
        for response_idx in range(1, response_count + 1):
            nested_value = nested.get(str(response_idx), nested.get(response_idx))
            score += _numeric_key_count(nested_value, rubric_count)

        if score > best_score:
            best_score = score
            best_nested = nested

    return best_nested


def _parse_batch_response_regex(text: str, expected_count: int, pi_mode: str) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    if pi_mode == "judge_prob":
        pattern = re.compile(
            r'"?(\d+)"?\s*:\s*\{\s*"criteria_met"\s*:\s*(true|false)\s*,\s*"prob_met"\s*:\s*([0-9]*\.?[0-9]+)\s*\}',
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            rubric_local_idx = int(match.group(1))
            if rubric_local_idx < 1 or rubric_local_idx > expected_count:
                continue
            normalized = _normalize_grading_result(
                {
                    "criteria_met": match.group(2).lower() == "true",
                    "prob_met": float(match.group(3)),
                },
                pi_mode=pi_mode,
            )
            if normalized is not None:
                results[rubric_local_idx] = normalized
        return results

    pattern = re.compile(
        r'"?(\d+)"?\s*:\s*\{\s*"criteria_met"\s*:\s*(true|false)\s*\}',
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        rubric_local_idx = int(match.group(1))
        if rubric_local_idx < 1 or rubric_local_idx > expected_count:
            continue
        normalized = _normalize_grading_result(
            {"criteria_met": match.group(2).lower() == "true"},
            pi_mode=pi_mode,
        )
        if normalized is not None:
            results[rubric_local_idx] = normalized

    return results


def _clamp_probability(value: Any, default: float = 0.0) -> float:
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, prob))


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _standard_healthbench_score(rubric_items: List[RubricItem], grading_results: List[Dict[str, Any]]) -> float:
    total_possible_points = sum(item.points for item in rubric_items if item.points > 0)
    if total_possible_points == 0:
        return 0.0

    achieved_points = sum(
        item.points
        for item, grading_result in zip(rubric_items, grading_results)
        if grading_result.get("criteria_met", False)
    )
    return achieved_points / total_possible_points


def _select_reported_acc(acc_mode: str, standard_score: float, aggregate: Any) -> bool:
    normalized_mode = (acc_mode or "standard").strip().lower()
    if normalized_mode in {"standard", "baseline", "healthbench"}:
        return standard_score > 0.5
    if normalized_mode == "flat":
        return aggregate.flat_reward > 0.5
    if normalized_mode == "hard":
        return aggregate.hard_reward > 0.5
    if normalized_mode == "dag":
        return aggregate.dag_reward > 0.5
    return aggregate.reward > 0.5


def _should_print_debug_summary() -> bool:
    global _reward_call_count
    with _reward_call_lock:
        _reward_call_count += 1
        call_idx = _reward_call_count
    return call_idx <= GEAR_DEBUG_PRINT_FIRST or (
        GEAR_DEBUG_PRINT_EVERY > 0 and call_idx % GEAR_DEBUG_PRINT_EVERY == 0
    )


def _normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    if prompt is None:
        return []
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(prompt, list):
        return []
    normalized = []
    for message in prompt:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        normalized.append({"role": role, "content": content})
    return normalized


def _format_conversation(prompt: List[Dict[str, str]], response: str) -> str:
    convo_with_response = prompt + [{"role": "assistant", "content": response}]
    return "\n\n".join(
        f"{message['role']}: {message['content']}"
        for message in convo_with_response
        if message.get("role") != "system"
    )


def _format_prompt_only_conversation(prompt: List[Dict[str, str]]) -> str:
    return "\n\n".join(
        f"{message['role']}: {message['content']}"
        for message in prompt
        if message.get("role") != "system"
    )


def _format_required_keys(count: int) -> str:
    return ", ".join(f'"{idx}"' for idx in range(1, count + 1))


def _build_batch_judge_prompt(
    prompt: List[Dict[str, str]],
    response: str,
    rubric_items: List[RubricItem],
    pi_mode: str,
) -> str:
    convo_str = _format_conversation(prompt, response)
    rubric_count = len(rubric_items)
    required_keys = _format_required_keys(rubric_count)
    rubric_items_str = "\n".join(
        f"{idx + 1}. (points={rubric_item.points}) {rubric_item.criterion}"
        for idx, rubric_item in enumerate(rubric_items)
    )
    template = JUDGE_TEMPLATE_BATCH_PROB if pi_mode == "judge_prob" else JUDGE_TEMPLATE_BATCH_BINARY
    return (
        template.replace("<<conversation>>", convo_str)
        .replace("<<rubric_items>>", rubric_items_str)
        .replace("<<rubric_count>>", str(rubric_count))
        .replace("<<required_keys>>", required_keys)
    )


def _build_multi_response_judge_prompt(
    prompt: List[Dict[str, str]],
    responses: List[str],
    rubric_items: List[RubricItem],
    pi_mode: str,
) -> str:
    convo_str = _format_prompt_only_conversation(prompt)
    response_count = len(responses)
    rubric_count = len(rubric_items)
    required_response_keys = _format_required_keys(response_count)
    required_rubric_keys = _format_required_keys(rubric_count)
    responses_str = "\n\n".join(
        f"Response {idx + 1}:\n{response}"
        for idx, response in enumerate(responses)
    )
    rubric_items_str = "\n".join(
        f"{idx + 1}. (points={rubric_item.points}) {rubric_item.criterion}"
        for idx, rubric_item in enumerate(rubric_items)
    )
    template = JUDGE_TEMPLATE_MULTI_RESPONSE_PROB if pi_mode == "judge_prob" else JUDGE_TEMPLATE_MULTI_RESPONSE_BINARY
    return (
        template.replace("<<conversation>>", convo_str)
        .replace("<<responses>>", responses_str)
        .replace("<<rubric_items>>", rubric_items_str)
        .replace("<<required_response_keys>>", required_response_keys)
        .replace("<<required_rubric_keys>>", required_rubric_keys)
    )


def _canonical_json(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = {
            str(key): _canonical_jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    elif isinstance(value, list):
        value = [_canonical_jsonable(item) for item in value]
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _canonical_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _canonical_jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonical_jsonable(item) for item in value]
    return value


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "tolist"):
        value = value.tolist()
    return _canonical_jsonable(value) if isinstance(value, dict) else {}


def _normalize_grading_result(grading: Dict[str, Any], pi_mode: str) -> Optional[Dict[str, Any]]:
    criteria_met = grading.get("criteria_met")
    if not isinstance(criteria_met, bool):
        return None

    prob_met = grading.get("prob_met")

    if pi_mode == "binary":
        return {
            "criteria_met": criteria_met,
            "prob_met": 1.0 if criteria_met else 0.0,
        }

    if prob_met is None:
        return None

    prob_met = _clamp_probability(prob_met, default=1.0 if criteria_met else 0.0)

    # Required internal semantics:
    #   prob_met = P(criteria_met is true)
    #
    # Some judge responses instead use prob_met as confidence in the stated
    # boolean. Example:
    #   {"criteria_met": false, "prob_met": 0.90}
    # usually means "90% confident that criteria_met is false".
    # Convert that to P(criteria_met=true)=0.10 before reward aggregation.
    if criteria_met and prob_met < 0.5:
        prob_met = 1.0 - prob_met
    elif not criteria_met and prob_met > 0.5:
        prob_met = 1.0 - prob_met

    prob_met = _clamp_probability(prob_met, default=1.0 if criteria_met else 0.0)

    return {
        "criteria_met": criteria_met,
        "prob_met": prob_met,
    }


def _parse_batch_grading_response(
    text: str,
    expected_count: int,
    pi_mode: str,
) -> BatchJudgeParseResult:
    parsed_payload: Optional[Dict[str, Any]] = None
    for candidate in _extract_json_candidates(text):
        try:
            raw_payload = json.loads(candidate)
        except json.JSONDecodeError:
            raw_payload = parse_json_to_dict(candidate)
        parsed_payload = _find_numeric_key_dict(raw_payload, expected_count)
        if parsed_payload is not None:
            break

    if parsed_payload is None:
        regex_results = _parse_batch_response_regex(text, expected_count, pi_mode)
        parse_status = "failed"
        if regex_results:
            parse_status = "partial" if len(regex_results) < expected_count else "full"
        return BatchJudgeParseResult(
            results=regex_results,
            parse_status=parse_status,
            missing_count=max(0, expected_count - len(regex_results)),
        )

    results: Dict[int, Dict[str, Any]] = {}
    for key, value in parsed_payload.items():
        rubric_local_idx = _parse_numeric_key(key, expected_count)
        if rubric_local_idx is None:
            continue

        if not isinstance(value, dict):
            continue

        normalized = _normalize_grading_result(value, pi_mode=pi_mode)
        if normalized is None:
            continue

        results[rubric_local_idx] = normalized

    parse_status = "failed"
    if results:
        parse_status = "partial" if len(results) < expected_count else "full"

    return BatchJudgeParseResult(
        results=results,
        parse_status=parse_status,
        missing_count=max(0, expected_count - len(results)),
    )


def _parse_multi_response_grading_response(
    text: str,
    response_count: int,
    rubric_count: int,
    pi_mode: str,
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], str, int]:
    parsed_payload: Optional[Dict[str, Any]] = None
    for candidate in _extract_json_candidates(text):
        try:
            raw_payload = json.loads(candidate)
        except json.JSONDecodeError:
            raw_payload = parse_json_to_dict(candidate)
        parsed_payload = _find_multi_response_dict(raw_payload, response_count, rubric_count)
        if parsed_payload is not None:
            break

    if parsed_payload is None:
        return {}, "failed", response_count * rubric_count

    results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for response_key, rubric_payload in parsed_payload.items():
        response_idx = _parse_numeric_key(response_key, response_count)
        if response_idx is None:
            continue
        if not isinstance(rubric_payload, dict):
            continue

        for rubric_key, value in rubric_payload.items():
            rubric_local_idx = _parse_numeric_key(rubric_key, rubric_count)
            if rubric_local_idx is None:
                continue

            if not isinstance(value, dict):
                continue

            normalized = _normalize_grading_result(value, pi_mode=pi_mode)
            if normalized is None:
                continue

            results.setdefault(response_idx, {})[rubric_local_idx] = normalized

    judged_count = sum(len(value) for value in results.values())
    expected_count = response_count * rubric_count
    parse_status = "failed"
    if judged_count:
        parse_status = "partial" if judged_count < expected_count else "full"

    return results, parse_status, max(0, expected_count - judged_count)


def _process_rule_task(task: Dict[str, Any], pi_mode: str) -> Dict[str, Any]:
    rubric_item: RubricItem = task["rubric_item"]
    function_name = rubric_item.tags.get("function")
    parameters = rubric_item.tags.get("parameters", {})
    verify_func = get_verification_function(function_name) if function_name else None
    if verify_func is None:
        return {
            "sample_idx": task["sample_idx"],
            "rubric_idx": task["rubric_idx"],
            "result": {
                "criteria_met": False,
                "prob_met": 0.0,
            },
        }

    criteria_met = bool(verify_func(task["response"], parameters))
    return {
        "sample_idx": task["sample_idx"],
        "rubric_idx": task["rubric_idx"],
        "result": {
            "criteria_met": criteria_met,
            "prob_met": 1.0 if criteria_met else 0.0,
        },
    }


def _default_missing_grading_result() -> Dict[str, Any]:
    return {
        "criteria_met": False,
        "prob_met": 0.0,
    }


def _maybe_raise_parse_failure(
    *,
    tag: str,
    sample_key: Any,
    parse_status: str,
    expected_count: int,
    missing_count: int,
    debug_paths: Dict[str, str],
) -> None:
    if parse_status == "full":
        return

    msg = (
        f"LLM judge parse failed: tag={tag}, sample_key={sample_key}, "
        f"parse_status={parse_status}, expected_count={expected_count}, "
        f"missing_count={missing_count}, debug_json={debug_paths.get('debug_json', '')}, "
        f"debug_txt={debug_paths.get('debug_txt', '')}, debug_dir={GEAR_DEBUG_DIR}, "
        f"VLLM_BASE_URL={os.getenv('VLLM_BASE_URL')}"
    )
    print("[GEAR][PARSE_FAILURE] " + msg, flush=True)

    if _fail_on_parse_error():
        raise RuntimeError(msg)


def _process_llm_task(task: Dict[str, Any], grader: VLLMSampler, pi_mode: str) -> Dict[str, Any]:
    sample_indices = task["sample_indices"]
    responses = task["responses"]
    rubric_count = len(task["rubric_items"])
    group_size = len(sample_indices)

    if group_size == 1:
        grader_prompt = _build_batch_judge_prompt(
            prompt=task["prompt"],
            response=responses[0],
            rubric_items=task["rubric_items"],
            pi_mode=pi_mode,
        )

        llm_results: Dict[int, Dict[str, Any]] = {}
        final_parse_status = "failed"
        final_missing_count = rubric_count
        retries_used = 0
        last_raw_response = ""
        last_response_metadata: Dict[str, Any] = {}
        last_debug_paths: Dict[str, str] = {}
        last_exception = ""

        for attempt_idx in range(3):
            retries_used = attempt_idx
            try:
                response = grader([{"role": "user", "content": grader_prompt}])
                last_raw_response = response.response_text or ""
                last_response_metadata = response.response_metadata or {}

                parse_result = _parse_batch_grading_response(
                    last_raw_response,
                    expected_count=rubric_count,
                    pi_mode=pi_mode,
                )
                llm_results = parse_result.results
                final_parse_status = parse_result.parse_status
                final_missing_count = parse_result.missing_count

                last_debug_paths = _write_debug_file(
                    tag="single",
                    sample_key=sample_indices[0],
                    attempt_idx=attempt_idx,
                    grader_prompt=grader_prompt,
                    raw_response=last_raw_response,
                    parse_status=final_parse_status,
                    missing_count=final_missing_count,
                    expected_count=rubric_count,
                    pi_mode=pi_mode,
                    response_metadata=last_response_metadata,
                    extra={
                        "sample_indices": sample_indices,
                        "rubric_count": rubric_count,
                    },
                    force=final_parse_status != "full",
                )

                if final_parse_status == "full":
                    break

            except Exception as exc:
                last_exception = repr(exc)
                last_debug_paths = _write_debug_file(
                    tag="single_exception",
                    sample_key=sample_indices[0],
                    attempt_idx=attempt_idx,
                    grader_prompt=grader_prompt,
                    raw_response=last_raw_response,
                    parse_status="exception",
                    missing_count=rubric_count,
                    expected_count=rubric_count,
                    pi_mode=pi_mode,
                    response_metadata=last_response_metadata,
                    exception=exc,
                    extra={
                        "sample_indices": sample_indices,
                        "rubric_count": rubric_count,
                    },
                    force=True,
                )
                if attempt_idx == 2:
                    raise

        _maybe_raise_parse_failure(
            tag="single",
            sample_key=sample_indices[0],
            parse_status=final_parse_status,
            expected_count=rubric_count,
            missing_count=final_missing_count,
            debug_paths=last_debug_paths,
        )

        results = []
        for local_idx, rubric_idx in enumerate(task["rubric_indices"], start=1):
            results.append(
                {
                    "sample_idx": sample_indices[0],
                    "rubric_idx": rubric_idx,
                    "result": llm_results.get(local_idx, _default_missing_grading_result()),
                }
            )

        debug_fields = _rollout_debug_fields(
            raw_response=last_raw_response,
            grader_prompt=grader_prompt,
            response_metadata=last_response_metadata,
            debug_paths=last_debug_paths,
            exception=last_exception,
        )

        return {
            "results": results,
            "stats": [
                {
                    "sample_idx": sample_indices[0],
                    "num_llm_rubrics": rubric_count,
                    "num_llm_judged": len(llm_results),
                    "num_llm_missing_judgments": final_missing_count,
                    "llm_parse_status": final_parse_status,
                    "llm_retries_used": retries_used,
                    "llm_group_size": group_size,
                    "llm_request_share": 1.0,
                    **debug_fields,
                }
            ],
        }

    grader_prompt = _build_multi_response_judge_prompt(
        prompt=task["prompt"],
        responses=responses,
        rubric_items=task["rubric_items"],
        pi_mode=pi_mode,
    )

    grouped_results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    final_parse_status = "failed"
    final_missing_count = group_size * rubric_count
    retries_used = 0
    last_raw_response = ""
    last_response_metadata: Dict[str, Any] = {}
    last_debug_paths: Dict[str, str] = {}
    last_exception = ""
    sample_key = "-".join(str(idx) for idx in sample_indices[:8])
    if len(sample_indices) > 8:
        sample_key += f"-plus{len(sample_indices) - 8}"

    for attempt_idx in range(3):
        retries_used = attempt_idx
        try:
            response = grader([{"role": "user", "content": grader_prompt}])
            last_raw_response = response.response_text or ""
            last_response_metadata = response.response_metadata or {}

            grouped_results, final_parse_status, final_missing_count = _parse_multi_response_grading_response(
                last_raw_response,
                response_count=group_size,
                rubric_count=rubric_count,
                pi_mode=pi_mode,
            )

            last_debug_paths = _write_debug_file(
                tag="multi",
                sample_key=sample_key,
                attempt_idx=attempt_idx,
                grader_prompt=grader_prompt,
                raw_response=last_raw_response,
                parse_status=final_parse_status,
                missing_count=final_missing_count,
                expected_count=group_size * rubric_count,
                pi_mode=pi_mode,
                response_metadata=last_response_metadata,
                extra={
                    "sample_indices": sample_indices,
                    "group_size": group_size,
                    "rubric_count": rubric_count,
                },
                force=final_parse_status != "full",
            )

            if final_parse_status == "full":
                break

        except Exception as exc:
            last_exception = repr(exc)
            last_debug_paths = _write_debug_file(
                tag="multi_exception",
                sample_key=sample_key,
                attempt_idx=attempt_idx,
                grader_prompt=grader_prompt,
                raw_response=last_raw_response,
                parse_status="exception",
                missing_count=group_size * rubric_count,
                expected_count=group_size * rubric_count,
                pi_mode=pi_mode,
                response_metadata=last_response_metadata,
                exception=exc,
                extra={
                    "sample_indices": sample_indices,
                    "group_size": group_size,
                    "rubric_count": rubric_count,
                },
                force=True,
            )
            if attempt_idx == 2:
                raise

    _maybe_raise_parse_failure(
        tag="multi",
        sample_key=sample_key,
        parse_status=final_parse_status,
        expected_count=group_size * rubric_count,
        missing_count=final_missing_count,
        debug_paths=last_debug_paths,
    )

    results = []
    stats = []
    debug_fields = _rollout_debug_fields(
        raw_response=last_raw_response,
        grader_prompt=grader_prompt,
        response_metadata=last_response_metadata,
        debug_paths=last_debug_paths,
        exception=last_exception,
    )

    for response_local_idx, sample_idx in enumerate(sample_indices, start=1):
        response_results = grouped_results.get(response_local_idx, {})
        for rubric_local_idx, rubric_idx in enumerate(task["rubric_indices"], start=1):
            results.append(
                {
                    "sample_idx": sample_idx,
                    "rubric_idx": rubric_idx,
                    "result": response_results.get(
                        rubric_local_idx,
                        _default_missing_grading_result(),
                    ),
                }
            )
        judged_count = len(response_results)
        response_parse_status = "failed"
        if judged_count:
            response_parse_status = "partial" if judged_count < rubric_count else "full"
        stats.append(
            {
                "sample_idx": sample_idx,
                "num_llm_rubrics": rubric_count,
                "num_llm_judged": judged_count,
                "num_llm_missing_judgments": max(0, rubric_count - judged_count),
                "llm_parse_status": response_parse_status if final_parse_status != "failed" else "failed",
                "llm_retries_used": retries_used,
                "llm_group_size": group_size,
                "llm_request_share": 1.0 / group_size,
                **debug_fields,
            }
        )

    return {"results": results, "stats": stats}


def _llm_group_key(prompt: List[Dict[str, str]], rubric_items: List[RubricItem], rubric_indices: List[int]) -> str:
    return _canonical_json(
        {
            "prompt": prompt,
            "rubric_indices": rubric_indices,
            "rubrics": [rubric_item.to_dict() for rubric_item in rubric_items],
        }
    )


def _build_tasks(batch_data: List[Tuple[str, str, str, Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rule_tasks: List[Dict[str, Any]] = []
    llm_tasks: List[Dict[str, Any]] = []
    active_llm_groups: Dict[str, Dict[str, Any]] = {}

    def new_llm_group(
        group_key: str,
        prompt: List[Dict[str, str]],
        rubric_items: List[RubricItem],
        rubric_indices: List[int],
    ) -> Dict[str, Any]:
        group = {
            "sample_indices": [],
            "prompt": prompt,
            "responses": [],
            "rubric_items": rubric_items,
            "rubric_indices": rubric_indices,
        }
        active_llm_groups[group_key] = group
        llm_tasks.append(group)
        return group

    for sample_idx, (_, solution_str, _, extra_info) in enumerate(batch_data):
        if not extra_info:
            continue
        prompt = _normalize_prompt(extra_info.get("prompt"))
        reward_model = _as_dict(extra_info.get("reward_model", {}))
        raw_rubrics = _as_list(reward_model.get("rubrics", []))
        if not prompt or not raw_rubrics:
            continue

        rubric_items = [RubricItem.from_dict(rubric, rubric_idx) for rubric_idx, rubric in enumerate(raw_rubrics)]
        llm_indices = []
        for rubric_idx, rubric_item in enumerate(rubric_items):
            if rubric_item.tags.get("verifier") == "rule" and rubric_item.tags.get("function"):
                rule_tasks.append(
                    {
                        "sample_idx": sample_idx,
                        "rubric_idx": rubric_idx,
                        "prompt": prompt,
                        "response": solution_str,
                        "rubric_item": rubric_item,
                    }
                )
            else:
                llm_indices.append(rubric_idx)

        if llm_indices:
            llm_rubric_items = [rubric_items[idx] for idx in llm_indices]
            max_responses_per_judge = max(1, GEAR_MAX_RESPONSES_PER_JUDGE)
            if GEAR_GROUP_RESPONSES_PER_PROMPT:
                group_key = _llm_group_key(prompt, llm_rubric_items, llm_indices)
                llm_group = active_llm_groups.get(group_key)
                if llm_group is None or len(llm_group["sample_indices"]) >= max_responses_per_judge:
                    llm_group = new_llm_group(
                        group_key=group_key,
                        prompt=prompt,
                        rubric_items=llm_rubric_items,
                        rubric_indices=llm_indices,
                    )
            else:
                llm_group = new_llm_group(
                    group_key=f"sample:{sample_idx}",
                    prompt=prompt,
                    rubric_items=llm_rubric_items,
                    rubric_indices=llm_indices,
                )
            llm_group["sample_indices"].append(sample_idx)
            llm_group["responses"].append(solution_str)

    return rule_tasks, llm_tasks


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    max_workers_per_url: int = MAX_CONCURRENT_WORKERS,
    aggregation_mode: str = "dag",
    pi_mode: str = "judge_prob",
    normalization_mode: str = "positive_sum",
    inference_mode: str = "approx",
    exact_if_num_nodes_le: int = 10,
    lambda_by_edge_type: Optional[Dict[str, float]] = None,
    graph_source: str = "dataset",
    acc_mode: str = "standard",
    **_: Any,
) -> List[Dict[str, Any]]:
    del graph_source

    batch_data = list(zip(data_sources, solution_strs, ground_truths, extra_infos))
    rule_tasks, llm_tasks = _build_tasks(batch_data)

    sample_results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    llm_task_stats: Dict[int, Dict[str, Any]] = {}
    for task in rule_tasks:
        result = _process_rule_task(task, pi_mode=pi_mode)
        sample_results.setdefault(result["sample_idx"], {})[result["rubric_idx"]] = result["result"]

    if llm_tasks:
        grader = get_global_grader()
        total_workers = max(1, min(len(llm_tasks), max_workers_per_url * max(1, len(grader.base_urls))))
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = [executor.submit(_process_llm_task, task, grader, pi_mode) for task in llm_tasks]
            for future in as_completed(futures):
                llm_task_output = future.result()
                for result in llm_task_output["results"]:
                    sample_results.setdefault(result["sample_idx"], {})[result["rubric_idx"]] = result["result"]
                for stats in llm_task_output["stats"]:
                    llm_task_stats[stats["sample_idx"]] = stats

    results: List[Dict[str, Any]] = []
    for sample_idx, (_, _, ground_truth, extra_info) in enumerate(batch_data):
        prompt = _normalize_prompt((extra_info or {}).get("prompt"))
        reward_model = _as_dict((extra_info or {}).get("reward_model", {}))
        raw_rubrics = _as_list(reward_model.get("rubrics", []))
        rubric_items = [RubricItem.from_dict(rubric, rubric_idx) for rubric_idx, rubric in enumerate(raw_rubrics)]
        sample_llm_stats = llm_task_stats.get(
            sample_idx,
            {
                "sample_idx": sample_idx,
                "num_llm_rubrics": 0,
                "num_llm_judged": 0,
                "num_llm_missing_judgments": 0,
                "llm_parse_status": "not_applicable",
                "llm_retries_used": 0,
                "llm_group_size": 1,
                "llm_request_share": 0.0,
            },
        )

        grading_results = []
        for rubric_idx in range(len(rubric_items)):
            grading_results.append(
                sample_results.get(sample_idx, {}).get(
                    rubric_idx,
                    _default_missing_grading_result(),
                )
            )

        if not prompt or not rubric_items:
            empty_result = {
                "score": 0.0,
                "acc": False,
                "ground_truth": ground_truth,
                "aggregation_mode": aggregation_mode,
                "baseline_reward": 0.0,
                "baseline_acc": False,
                "standard_score": 0.0,
                "standard_acc": False,
                "selected_reward": 0.0,
                "selected_acc": False,
                "flat_acc": False,
                "hard_acc": False,
                "dag_acc": False,
                "flat_reward": 0.0,
                "hard_reward": 0.0,
                "dag_reward": 0.0,
                "p_list": [],
                "q_list": [],
                "node_types": [],
                "graph_edges": [],
                "criteria_met_list": [],
                "rubric_ids": [],
                "num_graph_edges": 0,
                "num_prereq_edges": 0,
                "num_trigger_edges": 0,
                "has_graph_edges": 0.0,
                "num_gated_nodes_hard": 0,
                "num_suppressed_nodes_dag": 0,
                "mean_p": 0.0,
                "mean_q": 0.0,
                "mean_flat_q": 0.0,
                "mean_hard_q": 0.0,
                "mean_dag_q": 0.0,
                "reward_delta_flat_dag": 0.0,
                "reward_delta_flat_hard": 0.0,
                "reward_delta_dag_hard": 0.0,
                "aggregation_latency_ms": 0.0,
                "num_llm_rubrics": sample_llm_stats["num_llm_rubrics"],
                "num_llm_judged": sample_llm_stats["num_llm_judged"],
                "num_llm_missing_judgments": sample_llm_stats["num_llm_missing_judgments"],
                "llm_parse_status": sample_llm_stats["llm_parse_status"],
                "llm_retries_used": sample_llm_stats["llm_retries_used"],
                "llm_group_size": sample_llm_stats["llm_group_size"],
                "llm_request_share": sample_llm_stats["llm_request_share"],
            }
            if _debug_enabled():
                empty_result.update(
                    {
                        key: value
                        for key, value in sample_llm_stats.items()
                        if key.startswith("llm_")
                    }
                )
            results.append(empty_result)
            continue

        standard_score = _standard_healthbench_score(rubric_items, grading_results)
        aggregation_start_ns = time.perf_counter_ns()
        aggregate = aggregate_gear_reward(
            reward_model=reward_model,
            p_list=[item["prob_met"] for item in grading_results],
            criteria_met_list=[item["criteria_met"] for item in grading_results],
            aggregation_mode=aggregation_mode,
            normalization_mode=normalization_mode,
            inference_mode=inference_mode,
            exact_if_num_nodes_le=exact_if_num_nodes_le,
            lambda_by_edge_type=lambda_by_edge_type,
        )
        aggregation_latency_ms = (time.perf_counter_ns() - aggregation_start_ns) / 1_000_000.0

        graph_edges = aggregate.graph_edges
        num_graph_edges = len(graph_edges)
        num_trigger_edges = sum(1 for edge in graph_edges if edge["type"] == "trigger")
        num_prereq_edges = num_graph_edges - num_trigger_edges
        num_gated_nodes_hard = sum(
            1 for flat_q, hard_q in zip(aggregate.flat_q_list, aggregate.hard_q_list) if flat_q > 0.0 and hard_q == 0.0
        )
        num_suppressed_nodes_dag = sum(
            1 for flat_q, dag_q in zip(aggregate.flat_q_list, aggregate.dag_q_list) if flat_q - dag_q > 1e-6
        )

        result_item = {
            "score": aggregate.reward,
            "acc": _select_reported_acc(acc_mode=acc_mode, standard_score=standard_score, aggregate=aggregate),
            "ground_truth": ground_truth,
            "aggregation_mode": aggregate.aggregation_mode,
            "baseline_reward": standard_score,
            "baseline_acc": standard_score > 0.5,
            "standard_score": standard_score,
            "standard_acc": standard_score > 0.5,
            "selected_reward": aggregate.reward,
            "selected_acc": aggregate.reward > 0.5,
            "flat_acc": aggregate.flat_reward > 0.5,
            "hard_acc": aggregate.hard_reward > 0.5,
            "dag_acc": aggregate.dag_reward > 0.5,
            "flat_reward": aggregate.flat_reward,
            "hard_reward": aggregate.hard_reward,
            "dag_reward": aggregate.dag_reward,
            "p_list": aggregate.p_list,
            "q_list": aggregate.q_list,
            "flat_q_list": aggregate.flat_q_list,
            "hard_q_list": aggregate.hard_q_list,
            "dag_q_list": aggregate.dag_q_list,
            "node_types": aggregate.node_types,
            "graph_edges": graph_edges,
            "criteria_met_list": aggregate.criteria_met_list,
            "rubric_ids": aggregate.rubric_ids,
            "num_graph_edges": num_graph_edges,
            "num_prereq_edges": num_prereq_edges,
            "num_trigger_edges": num_trigger_edges,
            "has_graph_edges": float(num_graph_edges > 0),
            "num_gated_nodes_hard": num_gated_nodes_hard,
            "num_suppressed_nodes_dag": num_suppressed_nodes_dag,
            "mean_p": _mean(aggregate.p_list),
            "mean_q": _mean(aggregate.q_list),
            "mean_flat_q": _mean(aggregate.flat_q_list),
            "mean_hard_q": _mean(aggregate.hard_q_list),
            "mean_dag_q": _mean(aggregate.dag_q_list),
            "reward_delta_flat_dag": aggregate.flat_reward - aggregate.dag_reward,
            "reward_delta_flat_hard": aggregate.flat_reward - aggregate.hard_reward,
            "reward_delta_dag_hard": aggregate.dag_reward - aggregate.hard_reward,
            "aggregation_latency_ms": aggregation_latency_ms,
            "num_llm_rubrics": sample_llm_stats["num_llm_rubrics"],
            "num_llm_judged": sample_llm_stats["num_llm_judged"],
            "num_llm_missing_judgments": sample_llm_stats["num_llm_missing_judgments"],
            "llm_parse_status": sample_llm_stats["llm_parse_status"],
            "llm_retries_used": sample_llm_stats["llm_retries_used"],
            "llm_group_size": sample_llm_stats["llm_group_size"],
            "llm_request_share": sample_llm_stats["llm_request_share"],
        }

        if _debug_enabled():
            result_item.update(
                {
                    key: value
                    for key, value in sample_llm_stats.items()
                    if key.startswith("llm_")
                }
            )

        results.append(result_item)

    if results and _should_print_debug_summary():
        num_graph_edges_vals = [float(item.get("num_graph_edges", 0.0)) for item in results]
        num_gated_hard_vals = [float(item.get("num_gated_nodes_hard", 0.0)) for item in results]
        num_suppressed_dag_vals = [float(item.get("num_suppressed_nodes_dag", 0.0)) for item in results]
        reward_delta_flat_dag_vals = [float(item.get("reward_delta_flat_dag", 0.0)) for item in results]
        reward_delta_flat_hard_vals = [float(item.get("reward_delta_flat_hard", 0.0)) for item in results]
        has_graph_edges_vals = [float(item.get("has_graph_edges", 0.0)) for item in results]
        llm_missing_vals = [float(item.get("num_llm_missing_judgments", 0.0)) for item in results]
        llm_retry_vals = [float(item.get("llm_retries_used", 0.0)) for item in results]
        llm_group_size_vals = [float(item.get("llm_group_size", 1.0)) for item in results]
        llm_request_share_vals = [float(item.get("llm_request_share", 0.0)) for item in results]
        llm_parse_statuses = [str(item.get("llm_parse_status", "not_applicable")) for item in results]
        llm_parse_full_ratio = _mean([1.0 if status == "full" else 0.0 for status in llm_parse_statuses])
        llm_parse_partial_ratio = _mean([1.0 if status == "partial" else 0.0 for status in llm_parse_statuses])
        llm_parse_failed_ratio = _mean([1.0 if status == "failed" else 0.0 for status in llm_parse_statuses])
        print(
            "[GEAR_DEBUG] "
            f"mode={aggregation_mode} batch_size={len(results)} "
            f"graph_nonempty_ratio={_mean(has_graph_edges_vals):.3f} "
            f"avg_edges={_mean(num_graph_edges_vals):.3f} "
            f"avg_gated_hard={_mean(num_gated_hard_vals):.3f} "
            f"avg_suppressed_dag={_mean(num_suppressed_dag_vals):.3f} "
            f"avg_llm_missing={_mean(llm_missing_vals):.3f} "
            f"avg_llm_retries={_mean(llm_retry_vals):.3f} "
            f"avg_llm_group_size={_mean(llm_group_size_vals):.3f} "
            f"est_llm_requests={sum(llm_request_share_vals):.1f} "
            f"llm_parse_full_ratio={llm_parse_full_ratio:.3f} "
            f"llm_parse_partial_ratio={llm_parse_partial_ratio:.3f} "
            f"llm_parse_failed_ratio={llm_parse_failed_ratio:.3f} "
            f"avg_flat_minus_dag={_mean(reward_delta_flat_dag_vals):.6f} "
            f"avg_flat_minus_hard={_mean(reward_delta_flat_hard_vals):.6f} "
            f"debug_dir={GEAR_DEBUG_DIR}"
        )

    return results