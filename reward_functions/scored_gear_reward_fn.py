from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_BASE_SPEC = importlib_util.spec_from_file_location(
    "gear_batch_reward_base",
    Path(__file__).resolve().with_name("gear_batch_reward_fn.py"),
)
_BASE_MODULE = importlib_util.module_from_spec(_BASE_SPEC)
assert _BASE_SPEC.loader is not None
sys.modules[_BASE_SPEC.name] = _BASE_MODULE
_BASE_SPEC.loader.exec_module(_BASE_MODULE)

MAX_CONCURRENT_WORKERS = _BASE_MODULE.MAX_CONCURRENT_WORKERS
GEAR_GROUP_RESPONSES_PER_PROMPT = _BASE_MODULE.GEAR_GROUP_RESPONSES_PER_PROMPT
GEAR_MAX_RESPONSES_PER_JUDGE = _BASE_MODULE.GEAR_MAX_RESPONSES_PER_JUDGE

RubricItem = _BASE_MODULE.RubricItem
VLLMSampler = _BASE_MODULE.VLLMSampler
aggregate_gear_reward = _BASE_MODULE.aggregate_gear_reward
get_global_grader = _BASE_MODULE.get_global_grader
get_verification_function = _BASE_MODULE.get_verification_function

_as_dict = _BASE_MODULE._as_dict
_as_list = _BASE_MODULE._as_list
_canonical_json = _BASE_MODULE._canonical_json
_clamp_probability = _BASE_MODULE._clamp_probability
_debug_enabled = _BASE_MODULE._debug_enabled
_extract_json_candidates = _BASE_MODULE._extract_json_candidates
_find_multi_response_dict = _BASE_MODULE._find_multi_response_dict
_find_numeric_key_dict = _BASE_MODULE._find_numeric_key_dict
_format_conversation = _BASE_MODULE._format_conversation
_format_prompt_only_conversation = _BASE_MODULE._format_prompt_only_conversation
_format_required_keys = _BASE_MODULE._format_required_keys
_mean = _BASE_MODULE._mean
_maybe_raise_parse_failure = _BASE_MODULE._maybe_raise_parse_failure
_normalize_prompt = _BASE_MODULE._normalize_prompt
_parse_numeric_key = _BASE_MODULE._parse_numeric_key
_rollout_debug_fields = _BASE_MODULE._rollout_debug_fields
_select_reported_acc = _BASE_MODULE._select_reported_acc
_should_print_debug_summary = _BASE_MODULE._should_print_debug_summary
_write_debug_file = _BASE_MODULE._write_debug_file


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(str(os.getenv(name, str(default))).strip())
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, value)


# The base reward module exports response-level batching knobs, but this scored
# wrapper used to ignore GEAR_MAX_RUBRICS_PER_JUDGE. Keep it local here so a
# single judge request can be constrained by both response count and rubric count.
GEAR_MAX_RUBRICS_PER_JUDGE = _env_positive_int("GEAR_MAX_RUBRICS_PER_JUDGE", 4)

# Default to continue training on malformed judge output. Missing judgments are
# already converted to zero-score fallbacks below. Set this to 1 only when you
# want parse failures to abort the run for debugging.
GEAR_RAISE_PARSE_FAILURE = _env_flag("GEAR_RAISE_PARSE_FAILURE", False)

# Regex fallback salvages common Qwen3-8B malformed JSON cases, including missing
# object closers between top-level rubric keys.
GEAR_PARSE_REGEX_FALLBACK = _env_flag("GEAR_PARSE_REGEX_FALLBACK", True)


JUDGE_TEMPLATE_BATCH_SCORED = """
Your job is to evaluate the last assistant response in a conversation against multiple rubric items.

# Conversation
<<conversation>>

# Rubric Items
<<rubric_items>>

# Instructions
Return exactly one compact JSON object and nothing else.
Do not output markdown fences.
Do not output <think> or </think>.

Required rubric keys:
<<required_keys>>

The JSON object is invalid if any required key is missing.
The JSON object is invalid if it contains fewer than <<rubric_count>> rubric judgments.
Do not stop before returning all required keys.
Do not use rubric ids such as "r1".
Do not include explanations.
Do not include any text outside the JSON object.

For each rubric item:
- If score_type is "scale_1_10", assign a numeric "score" from 1 to 10 using the provided scoring rubric. Higher is better.
- If score_type is "partial_points", assign numeric "awarded_points" from 0 to max_points. Award partial credit according to the subitems in the rubric. Higher is better.

Example:
{
  "1": {"score": 8},
  "2": {"awarded_points": 15}
}
""".strip()


JUDGE_TEMPLATE_MULTI_RESPONSE_SCORED = """
Your job is to evaluate multiple candidate assistant responses to the same conversation against multiple rubric items.

# Conversation Before Candidate Responses
<<conversation>>

# Candidate Assistant Responses
<<responses>>

# Rubric Items
<<rubric_items>>

# Instructions
Return exactly one compact JSON object and nothing else.
Do not output markdown fences.
Do not output <think> or </think>.

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

For each rubric item:
- If score_type is "scale_1_10", assign a numeric "score" from 1 to 10 using the provided scoring rubric. Higher is better.
- If score_type is "partial_points", assign numeric "awarded_points" from 0 to max_points. Award partial credit according to the subitems in the rubric. Higher is better.

Example:
{
  "1": {
    "1": {"score": 8},
    "2": {"awarded_points": 15}
  },
  "2": {
    "1": {"score": 6},
    "2": {"awarded_points": 10}
  }
}
""".strip()


JUDGE_TEMPLATE_BATCH_SUBITEM_PROB = """
Your job is to evaluate the last assistant response in a conversation against multiple rubric items.

# Conversation
<<conversation>>

# Rubric Items and Scoring Subitems
<<rubric_items>>

# Instructions
Return exactly one compact JSON object and nothing else.
Do not output markdown fences.
Do not output <think> or </think>.

Required rubric keys:
<<required_keys>>

The JSON object is invalid if any required key is missing.
The JSON object is invalid if it contains fewer than <<rubric_count>> rubric judgments.
Do not stop before returning all required keys.
Do not use rubric ids such as "r1".
Do not include explanations.
Do not include any text outside the JSON object.

For each rubric item:
- Judge each listed subitem independently.
- Return "prob_met" for each subitem, a number from 0 to 1 indicating the probability that the candidate response satisfies that subitem.
- The reward code will multiply each subitem probability by its points and normalize by total points.

Example:
{
  "1": {"subitems": {"1": {"prob_met": 0.95}, "2": {"prob_met": 0.10}}},
  "2": {"subitems": {"1": {"prob_met": 0.80}}}
}
""".strip()


JUDGE_TEMPLATE_MULTI_RESPONSE_SUBITEM_PROB = """
Your job is to evaluate multiple candidate assistant responses to the same conversation against multiple rubric items.

# Conversation Before Candidate Responses
<<conversation>>

# Candidate Assistant Responses
<<responses>>

# Rubric Items and Scoring Subitems
<<rubric_items>>

# Instructions
Return exactly one compact JSON object and nothing else.
Do not output markdown fences.
Do not output <think> or </think>.

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

For each rubric item:
- Judge each listed subitem independently.
- Return "prob_met" for each subitem, a number from 0 to 1 indicating the probability that the candidate response satisfies that subitem.
- The reward code will multiply each subitem probability by its points and normalize by total points.

Example:
{
  "1": {
    "1": {"subitems": {"1": {"prob_met": 0.95}, "2": {"prob_met": 0.10}}},
    "2": {"subitems": {"1": {"prob_met": 0.80}}}
  },
  "2": {
    "1": {"subitems": {"1": {"prob_met": 0.20}, "2": {"prob_met": 0.30}}},
    "2": {"subitems": {"1": {"prob_met": 0.90}}}
  }
}
""".strip()


JUDGE_TEMPLATE_BATCH_BINARY_SUBITEM = """
Your job is to evaluate the last assistant response in a conversation against multiple rubric items.

# Conversation
<<conversation>>

# Rubric Items and Scoring Subitems
<<rubric_items>>

# Instructions
Return exactly one compact JSON object and nothing else.
Do not output markdown fences.
Do not output <think> or </think>.

Required rubric keys:
<<required_keys>>

The JSON object is invalid if any required key is missing.
The JSON object is invalid if it contains fewer than <<rubric_count>> rubric judgments.
Do not stop before returning all required keys.
Do not use rubric ids such as "r1".
Do not include explanations.
Do not include any text outside the JSON object.

For each rubric item:
- Judge each listed subitem independently.
- Return true if the candidate response satisfies that subitem, otherwise false.
- The reward code will add the points of satisfied subitems and assign 0 points to unsatisfied subitems.

Example:
{
  "1": {"subitems": {"1": true, "2": false}},
  "2": {"subitems": {"1": true}}
}
""".strip()


JUDGE_TEMPLATE_MULTI_RESPONSE_BINARY_SUBITEM = """
Your job is to evaluate multiple candidate assistant responses to the same conversation against multiple rubric items.

# Conversation Before Candidate Responses
<<conversation>>

# Candidate Assistant Responses
<<responses>>

# Rubric Items and Scoring Subitems
<<rubric_items>>

# Instructions
Return exactly one compact JSON object and nothing else.
Do not output markdown fences.
Do not output <think> or </think>.

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

For each rubric item:
- Judge each listed subitem independently.
- Return true if that candidate response satisfies that subitem, otherwise false.
- The reward code will add the points of satisfied subitems and assign 0 points to unsatisfied subitems.

Example:
{
  "1": {
    "1": {"subitems": {"1": true, "2": false}},
    "2": {"subitems": {"1": true}}
  },
  "2": {
    "1": {"subitems": {"1": false, "2": false}},
    "2": {"subitems": {"1": true}}
  }
}
""".strip()


@dataclass
class ScoreJudgeParseResult:
    results: Dict[int, Dict[str, Any]]
    parse_status: str
    missing_count: int


def _score_type_for(data_source: str, rubric_item: RubricItem) -> str:
    score_type = str(rubric_item.tags.get("score_type", "")).strip().lower()
    if score_type in {"scale_1_10", "partial_points"}:
        return score_type

    normalized_source = str(data_source).strip().lower()
    if normalized_source == "writingbench":
        return "scale_1_10"
    if normalized_source == "plawbench":
        return "partial_points"
    return "partial_points"


def _score_bounds(score_type: str, rubric_item: RubricItem) -> Tuple[float, float]:
    if score_type == "scale_1_10":
        return 1.0, 10.0
    return 0.0, max(float(rubric_item.points), 0.0)


def _clamp_number(value: Any, min_value: float, max_value: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, numeric))


def _effective_pi_mode(pi_mode: str) -> str:
    normalized = str(pi_mode or "judge_prob").strip().lower()
    if normalized == "binary":
        return "binary"
    if normalized in {"judge_prob", "prob", "normalized_score", "score", "soft"}:
        return "normalized_score"
    return "normalized_score"


def _is_plawbench(data_source: str) -> bool:
    return str(data_source).strip().lower() == "plawbench"


def _uses_subitem_prob(data_source: str, pi_mode: str) -> bool:
    return _is_plawbench(data_source) and _effective_pi_mode(pi_mode) == "normalized_score"


def _extract_point_subitems(criterion: str, max_points: float) -> List[Dict[str, Any]]:
    text = str(criterion or "")
    marker_pattern = re.compile(r"[\(（]\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*分?\s*[\)）]")
    matches = list(marker_pattern.finditer(text))
    if not matches:
        return [{"points": max_points, "text": text.strip()}]

    subitems = []
    for idx, match in enumerate(matches):
        points = _clamp_number(match.group(1), 0.0, max_points, 0.0)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        item_text = text[start:end].strip(" \n\t:：;；)）")
        if not item_text:
            prefix_start = max(0, match.start() - 80)
            item_text = text[prefix_start : match.start()].strip(" \n\t:：;；)）")
        subitems.append({"points": points, "text": item_text or text.strip()})

    if not subitems:
        return [{"points": max_points, "text": text.strip()}]
    return subitems


def _subitem_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, dict):
        for key in ("criteria_met", "met", "satisfied", "present", "fulfilled", "hit"):
            if isinstance(value.get(key), bool):
                return value[key]
            nested_bool = _subitem_bool(value.get(key))
            if nested_bool is not None:
                return nested_bool
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "present", "met", "satisfied", "fulfilled", "hit", "1", "是", "满足"}:
            return True
        if normalized in {"false", "no", "not_present", "not present", "unmet", "not_met", "0", "否", "不满足"}:
            return False
    return None


def _subitem_payload_get(payload: Any, idx: int) -> Any:
    if isinstance(payload, list):
        return payload[idx - 1] if idx - 1 < len(payload) else None
    if not isinstance(payload, dict):
        return None
    for key in (str(idx), idx, f"item_{idx}", f"subitem_{idx}", f"criterion_{idx}"):
        if key in payload:
            return payload[key]
    return None


def _subitem_probability(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        for key in ("prob_met", "probability", "prob", "p", "score"):
            if key in value:
                return _subitem_probability(value[key])
    criteria_met = _subitem_bool(value)
    if criteria_met is not None:
        return 1.0 if criteria_met else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _clamp_probability(value, default=0.0)
    if isinstance(value, str):
        try:
            return _clamp_probability(float(value.strip()), default=0.0)
        except ValueError:
            return None
    return None


def _score_from_binary_subitems(raw: Dict[str, Any], rubric_item: RubricItem, max_score: float) -> Optional[Dict[str, Any]]:
    subitems = _extract_point_subitems(rubric_item.criterion, max_score)
    payload = raw.get("subitems")
    if payload is None:
        payload = raw.get("items")
    if payload is None:
        payload = raw.get("criteria")
    if not isinstance(payload, (dict, list)):
        criteria_met = _subitem_bool(raw.get("criteria_met"))
        if criteria_met is None:
            return None
        awarded = max_score if criteria_met else 0.0
        return {
            "awarded_score": awarded,
            "subitem_results": [
                {
                    "index": 1,
                    "points": max_score,
                    "criteria_met": bool(criteria_met),
                    "text": rubric_item.criterion,
                }
            ],
        }

    awarded = 0.0
    subitem_results = []
    for idx, subitem in enumerate(subitems, start=1):
        criteria_met = _subitem_bool(_subitem_payload_get(payload, idx))
        if criteria_met is None:
            criteria_met = False
        points = float(subitem["points"])
        if criteria_met:
            awarded += points
        subitem_results.append(
            {
                "index": idx,
                "points": points,
                "criteria_met": bool(criteria_met),
                "text": subitem["text"],
            }
        )

    return {
        "awarded_score": _clamp_number(awarded, 0.0, max_score, 0.0),
        "subitem_results": subitem_results,
    }


def _score_from_prob_subitems(raw: Dict[str, Any], rubric_item: RubricItem, max_score: float) -> Optional[Dict[str, Any]]:
    subitems = _extract_point_subitems(rubric_item.criterion, max_score)
    payload = raw.get("subitems")
    if payload is None:
        payload = raw.get("items")
    if payload is None:
        payload = raw.get("criteria")
    if not isinstance(payload, (dict, list)):
        return None

    awarded = 0.0
    subitem_results = []
    for idx, subitem in enumerate(subitems, start=1):
        prob_met = _subitem_probability(_subitem_payload_get(payload, idx))
        if prob_met is None:
            prob_met = 0.0
        points = float(subitem["points"])
        awarded += prob_met * points
        subitem_results.append(
            {
                "index": idx,
                "points": points,
                "prob_met": prob_met,
                "criteria_met": prob_met >= 0.5,
                "text": subitem["text"],
            }
        )

    return {
        "awarded_score": _clamp_number(awarded, 0.0, max_score, 0.0),
        "subitem_results": subitem_results,
    }


def _direct_awarded_score(raw: Dict[str, Any], max_score: float) -> Optional[float]:
    for key in ("normalized_score", "prob_met", "probability", "prob", "p"):
        if key in raw:
            return _clamp_probability(raw[key], default=0.0) * max_score

    criteria_met = _subitem_bool(raw.get("criteria_met"))
    if criteria_met is not None:
        return max_score if criteria_met else 0.0

    return None


def _fallback_raw_score(raw: Dict[str, Any], score_type: str, min_score: float, max_score: float) -> float:
    if score_type == "scale_1_10":
        raw_score = raw.get("score", raw.get("awarded_points"))
        default = min_score
    else:
        raw_score = raw.get("awarded_points", raw.get("score"))
        default = min_score
    return _clamp_number(raw_score, min_score, max_score, default)


def _coerce_scored_payload_value(
    value: Any,
    *,
    data_source: str,
    rubric_item: RubricItem,
    pi_mode: str,
) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value

    if isinstance(value, bool):
        return {"criteria_met": value}

    score_type = _score_type_for(data_source, rubric_item)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if _effective_pi_mode(pi_mode) == "binary":
            return {"criteria_met": bool(numeric)}
        if _uses_subitem_prob(data_source, pi_mode) and 0.0 <= numeric <= 1.0:
            return {"prob_met": numeric}
        if score_type == "scale_1_10":
            return {"score": numeric}
        return {"awarded_points": numeric}

    if isinstance(value, str):
        criteria_met = _subitem_bool(value)
        if criteria_met is not None:
            return {"criteria_met": criteria_met}
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
        return _coerce_scored_payload_value(
            numeric,
            data_source=data_source,
            rubric_item=rubric_item,
            pi_mode=pi_mode,
        )

    return None


def _top_level_list_payload(raw_payload: Any, expected_count: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_payload, list) or not raw_payload:
        return None
    return {
        str(idx + 1): value
        for idx, value in enumerate(raw_payload[:expected_count])
    }


def _strip_think_blocks(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def _append_missing_json_closers(text: str) -> str:
    """Append missing closing braces/brackets at the tail of a JSON-like string.

    This only repairs tail truncation. Mid-object omissions are handled by the
    subitem regex fallback below.
    """
    s = str(text or "").strip()
    if not s:
        return s

    stack: List[str] = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        s += '"'
    if stack:
        s += "".join(reversed(stack))
    return s


def _json_candidate_variants(text: str) -> List[str]:
    cleaned = _strip_think_blocks(text)
    candidates: List[str] = []

    # Keep the base extractor first; it should parse correct JSON without any
    # extra heuristics.
    candidates.extend(_extract_json_candidates(cleaned))

    # Markdown-fenced JSON sometimes slips through despite the prompt.
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL):
        candidates.append(match.group(1).strip())

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj >= 0:
        if last_obj > first_obj:
            candidates.append(cleaned[first_obj : last_obj + 1])
        candidates.append(_append_missing_json_closers(cleaned[first_obj:]))

    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _parse_batch_subitem_response_regex(
    text: str,
    expected_count: int,
    *,
    data_source: str,
    rubric_items: List[RubricItem],
    met_threshold: float,
    pi_mode: str,
) -> Dict[int, Dict[str, Any]]:
    """Salvage subitem-form judge output when the top-level JSON is malformed.

    This is intentionally conservative: it only activates for the two subitem
    modes used by this file. It scans rubric headers of the form
    "1": {"subitems": {...}} and stops each rubric section at the next such
    header. This handles the observed Qwen3 failure mode where the model emits
    all rubric sections but forgets one or more closing braces between sections.
    """
    if not GEAR_PARSE_REGEX_FALLBACK:
        return {}

    effective_pi_mode = _effective_pi_mode(pi_mode)
    if effective_pi_mode != "binary" and not _uses_subitem_prob(data_source, pi_mode):
        return {}

    cleaned = _strip_think_blocks(text)
    header_pattern = re.compile(
        r'"?(\d+)"?\s*:\s*\{\s*"subitems"\s*:\s*\{',
        flags=re.IGNORECASE,
    )
    headers = []
    for match in header_pattern.finditer(cleaned):
        rubric_local_idx = _parse_numeric_key(match.group(1), expected_count)
        if rubric_local_idx is None:
            continue
        headers.append((rubric_local_idx, match.start(), match.end()))

    if not headers:
        return {}

    prob_pattern = re.compile(
        r'"?(\d+)"?\s*:\s*\{\s*"prob_met"\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\}',
        flags=re.IGNORECASE,
    )
    bool_pattern = re.compile(
        r'"?(\d+)"?\s*:\s*(true|false)\b',
        flags=re.IGNORECASE,
    )

    results: Dict[int, Dict[str, Any]] = {}
    for header_idx, (rubric_local_idx, _, payload_start) in enumerate(headers):
        section_end = headers[header_idx + 1][1] if header_idx + 1 < len(headers) else len(cleaned)
        section = cleaned[payload_start:section_end]

        subitems_payload: Dict[str, Any] = {}
        if effective_pi_mode == "binary":
            for match in bool_pattern.finditer(section):
                subitems_payload[match.group(1)] = match.group(2).lower() == "true"
        else:
            for match in prob_pattern.finditer(section):
                try:
                    value = float(match.group(2))
                except ValueError:
                    continue
                subitems_payload[match.group(1)] = {"prob_met": value}

        if not subitems_payload:
            continue

        normalized = _normalize_scored_result(
            {"subitems": subitems_payload},
            data_source=data_source,
            rubric_item=rubric_items[rubric_local_idx - 1],
            met_threshold=met_threshold,
            pi_mode=pi_mode,
        )
        if normalized is not None:
            results[rubric_local_idx] = normalized

    return results


def _maybe_raise_or_warn_parse_failure(
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

    if GEAR_RAISE_PARSE_FAILURE:
        _maybe_raise_parse_failure(
            tag=tag,
            sample_key=sample_key,
            parse_status=parse_status,
            expected_count=expected_count,
            missing_count=missing_count,
            debug_paths=debug_paths,
        )
        return

    print(
        "[SCORED_GEAR][WARN_PARSE_FAILURE_CONTINUE] "
        f"tag={tag}, sample_key={sample_key}, parse_status={parse_status}, "
        f"expected_count={expected_count}, missing_count={missing_count}, "
        f"debug_json={(debug_paths or {}).get('debug_json', '')}, "
        f"debug_txt={(debug_paths or {}).get('debug_txt', '')}",
        flush=True,
    )


def _normalize_scored_result(
    raw: Dict[str, Any],
    *,
    data_source: str,
    rubric_item: RubricItem,
    met_threshold: float,
    pi_mode: str,
) -> Optional[Dict[str, Any]]:
    score_type = _score_type_for(data_source, rubric_item)
    min_score, max_score = _score_bounds(score_type, rubric_item)
    if max_score <= 0:
        return None

    subitem_results = []
    if _effective_pi_mode(pi_mode) == "binary":
        binary_score = _score_from_binary_subitems(raw, rubric_item, max_score)
        if binary_score is not None:
            awarded_score = binary_score["awarded_score"]
            subitem_results = binary_score["subitem_results"]
        else:
            direct_score = _direct_awarded_score(raw, max_score)
            awarded_score = (
                direct_score
                if direct_score is not None
                else _fallback_raw_score(raw, score_type, min_score, max_score)
            )
    elif _uses_subitem_prob(data_source, pi_mode):
        prob_score = _score_from_prob_subitems(raw, rubric_item, max_score)
        if prob_score is not None:
            awarded_score = prob_score["awarded_score"]
            subitem_results = prob_score["subitem_results"]
        else:
            direct_score = _direct_awarded_score(raw, max_score)
            awarded_score = (
                direct_score
                if direct_score is not None
                else _fallback_raw_score(raw, score_type, min_score, max_score)
            )
    else:
        direct_score = _direct_awarded_score(raw, max_score)
        awarded_score = (
            direct_score
            if direct_score is not None
            else _fallback_raw_score(raw, score_type, min_score, max_score)
        )

    normalized_score = _clamp_probability(awarded_score / max_score, default=0.0)
    criteria_met = normalized_score >= met_threshold
    return {
        "criteria_met": criteria_met,
        "prob_met": normalized_score,
        "normalized_score": normalized_score,
        "awarded_score": awarded_score,
        "max_score": max_score,
        "score_type": score_type,
        "subitem_results": subitem_results,
    }


def _parse_batch_response_regex(
    text: str,
    expected_count: int,
    *,
    data_source: str,
    rubric_items: List[RubricItem],
    met_threshold: float,
    pi_mode: str,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    key_pattern = r"(?:\d+|r[_\-\s]*\d+|rubric[_\-\s]*\d+|item[_\-\s]*\d+|criterion[_\-\s]*\d+)"
    pattern = re.compile(
        rf'"?({key_pattern})"?\s*:\s*\{{[^{{}}]*(?:"score"|"awarded_points")\s*:\s*([0-9]*\.?[0-9]+)[^{{}}]*\}}',
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        rubric_local_idx = _parse_numeric_key(match.group(1), expected_count)
        if rubric_local_idx is None:
            continue
        raw = {"score": float(match.group(2))}
        normalized = _normalize_scored_result(
            raw,
            data_source=data_source,
            rubric_item=rubric_items[rubric_local_idx - 1],
            met_threshold=met_threshold,
            pi_mode=pi_mode,
        )
        if normalized is not None:
            results[rubric_local_idx] = normalized
    return results


def _parse_batch_scored_response(
    text: str,
    *,
    expected_count: int,
    data_source: str,
    rubric_items: List[RubricItem],
    met_threshold: float,
    pi_mode: str,
) -> ScoreJudgeParseResult:
    parsed_payload: Optional[Dict[str, Any]] = None
    for candidate in _json_candidate_variants(text):
        try:
            raw_payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed_payload = _find_numeric_key_dict(raw_payload, expected_count)
        if parsed_payload is None:
            parsed_payload = _top_level_list_payload(raw_payload, expected_count)
        if parsed_payload is not None:
            break

    if parsed_payload is None:
        regex_results = _parse_batch_subitem_response_regex(
            text,
            expected_count,
            data_source=data_source,
            rubric_items=rubric_items,
            met_threshold=met_threshold,
            pi_mode=pi_mode,
        )
        if not regex_results:
            regex_results = _parse_batch_response_regex(
                text,
                expected_count,
                data_source=data_source,
                rubric_items=rubric_items,
                met_threshold=met_threshold,
                pi_mode=pi_mode,
            )
        parse_status = "failed"
        if regex_results:
            parse_status = "partial" if len(regex_results) < expected_count else "full"
        return ScoreJudgeParseResult(
            results=regex_results,
            parse_status=parse_status,
            missing_count=max(0, expected_count - len(regex_results)),
        )

    results: Dict[int, Dict[str, Any]] = {}
    for key, value in parsed_payload.items():
        rubric_local_idx = _parse_numeric_key(key, expected_count)
        if rubric_local_idx is None:
            continue
        coerced_value = _coerce_scored_payload_value(
            value,
            data_source=data_source,
            rubric_item=rubric_items[rubric_local_idx - 1],
            pi_mode=pi_mode,
        )
        if coerced_value is None:
            continue
        normalized = _normalize_scored_result(
            coerced_value,
            data_source=data_source,
            rubric_item=rubric_items[rubric_local_idx - 1],
            met_threshold=met_threshold,
            pi_mode=pi_mode,
        )
        if normalized is not None:
            results[rubric_local_idx] = normalized

    parse_status = "failed"
    if results:
        parse_status = "partial" if len(results) < expected_count else "full"

    return ScoreJudgeParseResult(
        results=results,
        parse_status=parse_status,
        missing_count=max(0, expected_count - len(results)),
    )


def _parse_multi_response_scored_response(
    text: str,
    *,
    response_count: int,
    rubric_count: int,
    data_source: str,
    rubric_items: List[RubricItem],
    met_threshold: float,
    pi_mode: str,
) -> Tuple[Dict[int, Dict[int, Dict[str, Any]]], str, int]:
    parsed_payload: Optional[Dict[str, Any]] = None
    for candidate in _json_candidate_variants(text):
        try:
            raw_payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        parsed_payload = _find_multi_response_dict(raw_payload, response_count, rubric_count)
        if parsed_payload is None:
            parsed_payload = _top_level_list_payload(raw_payload, response_count)
        if parsed_payload is not None:
            break

    if parsed_payload is None:
        return {}, "failed", response_count * rubric_count

    results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for response_key, rubric_payload in parsed_payload.items():
        response_idx = _parse_numeric_key(response_key, response_count)
        if response_idx is None:
            continue
        if isinstance(rubric_payload, list):
            rubric_payload = _top_level_list_payload(rubric_payload, rubric_count)
        if not isinstance(rubric_payload, dict):
            continue

        for rubric_key, value in rubric_payload.items():
            rubric_local_idx = _parse_numeric_key(rubric_key, rubric_count)
            if rubric_local_idx is None:
                continue
            coerced_value = _coerce_scored_payload_value(
                value,
                data_source=data_source,
                rubric_item=rubric_items[rubric_local_idx - 1],
                pi_mode=pi_mode,
            )
            if coerced_value is None:
                continue
            normalized = _normalize_scored_result(
                coerced_value,
                data_source=data_source,
                rubric_item=rubric_items[rubric_local_idx - 1],
                met_threshold=met_threshold,
                pi_mode=pi_mode,
            )
            if normalized is not None:
                results.setdefault(response_idx, {})[rubric_local_idx] = normalized

    judged_count = sum(len(value) for value in results.values())
    expected_count = response_count * rubric_count
    parse_status = "failed"
    if judged_count:
        parse_status = "partial" if judged_count < expected_count else "full"

    return results, parse_status, max(0, expected_count - judged_count)


def _format_rubric_items(data_source: str, rubric_items: List[RubricItem], pi_mode: str) -> str:
    lines = []
    for idx, rubric_item in enumerate(rubric_items):
        score_type = _score_type_for(data_source, rubric_item)
        _, max_score = _score_bounds(score_type, rubric_item)
        if _effective_pi_mode(pi_mode) == "binary" or _uses_subitem_prob(data_source, pi_mode):
            lines.append(
                f"{idx + 1}. (score_type={score_type}, max_points={max_score}, weight={rubric_item.points})"
            )
            lines.append("Subitems:")
            for subitem_idx, subitem in enumerate(_extract_point_subitems(rubric_item.criterion, max_score), start=1):
                subitem_text = " ".join(str(subitem["text"]).split())
                lines.append(f"  {subitem_idx}. (+{float(subitem['points']):g}) {subitem_text}")
            continue
        lines.append(
            f"{idx + 1}. (score_type={score_type}, max_points={max_score}, weight={rubric_item.points}) "
            f"{rubric_item.criterion}"
        )
    return "\n".join(lines)


def _build_batch_scored_prompt(
    *,
    data_source: str,
    prompt: List[Dict[str, str]],
    response: str,
    rubric_items: List[RubricItem],
    pi_mode: str,
) -> str:
    rubric_count = len(rubric_items)
    if _effective_pi_mode(pi_mode) == "binary":
        template = JUDGE_TEMPLATE_BATCH_BINARY_SUBITEM
    elif _uses_subitem_prob(data_source, pi_mode):
        template = JUDGE_TEMPLATE_BATCH_SUBITEM_PROB
    else:
        template = JUDGE_TEMPLATE_BATCH_SCORED
    return (
        template.replace("<<conversation>>", _format_conversation(prompt, response))
        .replace("<<rubric_items>>", _format_rubric_items(data_source, rubric_items, pi_mode))
        .replace("<<rubric_count>>", str(rubric_count))
        .replace("<<required_keys>>", _format_required_keys(rubric_count))
    )


def _build_multi_response_scored_prompt(
    *,
    data_source: str,
    prompt: List[Dict[str, str]],
    responses: List[str],
    rubric_items: List[RubricItem],
    pi_mode: str,
) -> str:
    response_count = len(responses)
    rubric_count = len(rubric_items)
    responses_str = "\n\n".join(
        f"Response {idx + 1}:\n{response}"
        for idx, response in enumerate(responses)
    )
    if _effective_pi_mode(pi_mode) == "binary":
        template = JUDGE_TEMPLATE_MULTI_RESPONSE_BINARY_SUBITEM
    elif _uses_subitem_prob(data_source, pi_mode):
        template = JUDGE_TEMPLATE_MULTI_RESPONSE_SUBITEM_PROB
    else:
        template = JUDGE_TEMPLATE_MULTI_RESPONSE_SCORED
    return (
        template.replace("<<conversation>>", _format_prompt_only_conversation(prompt))
        .replace("<<responses>>", responses_str)
        .replace("<<rubric_items>>", _format_rubric_items(data_source, rubric_items, pi_mode))
        .replace("<<required_response_keys>>", _format_required_keys(response_count))
        .replace("<<required_rubric_keys>>", _format_required_keys(rubric_count))
    )


def _default_missing_scored_result(rubric_item: RubricItem, data_source: str) -> Dict[str, Any]:
    score_type = _score_type_for(data_source, rubric_item)
    _, max_score = _score_bounds(score_type, rubric_item)
    return {
        "criteria_met": False,
        "prob_met": 0.0,
        "normalized_score": 0.0,
        "awarded_score": 0.0,
        "max_score": max_score,
        "score_type": score_type,
        "subitem_results": [],
    }


def _write_scored_debug_file(**kwargs: Any) -> Dict[str, str]:
    try:
        return _write_debug_file(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'force'" not in str(exc):
            raise
        kwargs.pop("force", None)
        return _write_debug_file(**kwargs)


def _print_parse_failure_context(
    *,
    tag: str,
    sample_key: Any,
    parse_status: str,
    expected_count: int,
    missing_count: int,
    raw_response: str,
    grader_prompt: str,
    response_metadata: Dict[str, Any],
    debug_paths: Dict[str, str],
) -> None:
    if parse_status == "full":
        return

    try:
        raw_limit = int(os.getenv("GEAR_DEBUG_RAW_RESPONSE_CHARS", "12000"))
    except ValueError:
        raw_limit = 12000
    raw_limit = max(0, raw_limit)
    raw_preview = (raw_response or "")[:raw_limit]

    print(
        "[SCORED_GEAR][PARSE_FAILURE_CONTEXT] "
        f"tag={tag}, sample_key={sample_key}, parse_status={parse_status}, "
        f"expected_count={expected_count}, missing_count={missing_count}, "
        f"raw_response_len={len(raw_response or '')}, grader_prompt_len={len(grader_prompt or '')}, "
        f"response_metadata={json.dumps(response_metadata or {}, ensure_ascii=False, default=str)}, "
        f"debug_json={(debug_paths or {}).get('debug_json', '')}, "
        f"debug_txt={(debug_paths or {}).get('debug_txt', '')}",
        flush=True,
    )
    print(
        "[SCORED_GEAR][RAW_RESPONSE_BEGIN]\n"
        + raw_preview
        + "\n[SCORED_GEAR][RAW_RESPONSE_END]",
        flush=True,
    )


def _process_rule_task(task: Dict[str, Any]) -> Dict[str, Any]:
    rubric_item: RubricItem = task["rubric_item"]
    function_name = rubric_item.tags.get("function")
    parameters = rubric_item.tags.get("parameters", {})
    verify_func = get_verification_function(function_name) if function_name else None
    if verify_func is None:
        normalized_score = 0.0
    else:
        normalized_score = 1.0 if bool(verify_func(task["response"], parameters)) else 0.0

    score_type = _score_type_for(task["data_source"], rubric_item)
    _, max_score = _score_bounds(score_type, rubric_item)
    criteria_met = normalized_score >= task["met_threshold"]
    return {
        "sample_idx": task["sample_idx"],
        "rubric_idx": task["rubric_idx"],
        "result": {
            "criteria_met": criteria_met,
            "prob_met": normalized_score,
            "normalized_score": normalized_score,
            "awarded_score": normalized_score * max_score,
            "max_score": max_score,
            "score_type": score_type,
            "subitem_results": [],
        },
    }


def _process_llm_task(task: Dict[str, Any], grader: VLLMSampler, met_threshold: float, pi_mode: str) -> Dict[str, Any]:
    sample_indices = task["sample_indices"]
    responses = task["responses"]
    rubric_items = task["rubric_items"]
    rubric_count = len(rubric_items)
    group_size = len(sample_indices)
    data_source = task["data_source"]

    if group_size == 1:
        grader_prompt = _build_batch_scored_prompt(
            data_source=data_source,
            prompt=task["prompt"],
            response=responses[0],
            rubric_items=rubric_items,
            pi_mode=pi_mode,
        )
        final_results: Dict[int, Dict[str, Any]] = {}
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
                parse_result = _parse_batch_scored_response(
                    last_raw_response,
                    expected_count=rubric_count,
                    data_source=data_source,
                    rubric_items=rubric_items,
                    met_threshold=met_threshold,
                    pi_mode=pi_mode,
                )
                final_results = parse_result.results
                final_parse_status = parse_result.parse_status
                final_missing_count = parse_result.missing_count
                last_debug_paths = _write_scored_debug_file(
                    tag="scored_single",
                    sample_key=sample_indices[0],
                    attempt_idx=attempt_idx,
                    grader_prompt=grader_prompt,
                    raw_response=last_raw_response,
                    parse_status=final_parse_status,
                    missing_count=final_missing_count,
                    expected_count=rubric_count,
                    pi_mode=_effective_pi_mode(pi_mode),
                    response_metadata=last_response_metadata,
                    extra={"data_source": data_source, "rubric_count": rubric_count},
                    force=final_parse_status != "full",
                )
                if final_parse_status == "full":
                    break
            except Exception as exc:  # noqa: BLE001
                last_exception = repr(exc)
                last_debug_paths = _write_scored_debug_file(
                    tag="scored_single_exception",
                    sample_key=sample_indices[0],
                    attempt_idx=attempt_idx,
                    grader_prompt=grader_prompt,
                    raw_response=last_raw_response,
                    parse_status="exception",
                    missing_count=rubric_count,
                    expected_count=rubric_count,
                    pi_mode=_effective_pi_mode(pi_mode),
                    response_metadata=last_response_metadata,
                    exception=exc,
                    extra={"data_source": data_source, "rubric_count": rubric_count},
                    force=True,
                )
                if attempt_idx == 2:
                    raise

        _print_parse_failure_context(
            tag="scored_single",
            sample_key=sample_indices[0],
            parse_status=final_parse_status,
            expected_count=rubric_count,
            missing_count=final_missing_count,
            raw_response=last_raw_response,
            grader_prompt=grader_prompt,
            response_metadata=last_response_metadata,
            debug_paths=last_debug_paths,
        )
        _maybe_raise_or_warn_parse_failure(
            tag="scored_single",
            sample_key=sample_indices[0],
            parse_status=final_parse_status,
            expected_count=rubric_count,
            missing_count=final_missing_count,
            debug_paths=last_debug_paths,
        )

        debug_fields = _rollout_debug_fields(
            raw_response=last_raw_response,
            grader_prompt=grader_prompt,
            response_metadata=last_response_metadata,
            debug_paths=last_debug_paths,
            exception=last_exception,
        )
        results = [
            {
                "sample_idx": sample_indices[0],
                "rubric_idx": rubric_idx,
                "result": final_results.get(
                    rubric_local_idx,
                    _default_missing_scored_result(rubric_items[rubric_local_idx - 1], data_source),
                ),
            }
            for rubric_local_idx, rubric_idx in enumerate(task["rubric_indices"], start=1)
        ]
        judged_count = len(final_results)
        parse_status = "failed"
        if judged_count:
            parse_status = "partial" if judged_count < rubric_count else "full"
        return {
            "results": results,
            "stats": [
                {
                    "sample_idx": sample_indices[0],
                    "num_llm_rubrics": rubric_count,
                    "num_llm_judged": judged_count,
                    "num_llm_missing_judgments": max(0, rubric_count - judged_count),
                    "llm_parse_status": parse_status,
                    "llm_retries_used": retries_used,
                    "llm_group_size": 1,
                    "llm_request_share": 1.0,
                    **debug_fields,
                }
            ],
        }

    grader_prompt = _build_multi_response_scored_prompt(
        data_source=data_source,
        prompt=task["prompt"],
        responses=responses,
        rubric_items=rubric_items,
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
    sample_key = ",".join(str(idx) for idx in sample_indices)

    for attempt_idx in range(3):
        retries_used = attempt_idx
        try:
            response = grader([{"role": "user", "content": grader_prompt}])
            last_raw_response = response.response_text or ""
            last_response_metadata = response.response_metadata or {}
            grouped_results, final_parse_status, final_missing_count = _parse_multi_response_scored_response(
                last_raw_response,
                response_count=group_size,
                rubric_count=rubric_count,
                data_source=data_source,
                rubric_items=rubric_items,
                met_threshold=met_threshold,
                pi_mode=pi_mode,
            )
            last_debug_paths = _write_scored_debug_file(
                tag="scored_multi",
                sample_key=sample_key,
                attempt_idx=attempt_idx,
                grader_prompt=grader_prompt,
                raw_response=last_raw_response,
                parse_status=final_parse_status,
                missing_count=final_missing_count,
                expected_count=group_size * rubric_count,
                pi_mode=_effective_pi_mode(pi_mode),
                response_metadata=last_response_metadata,
                extra={"data_source": data_source, "group_size": group_size, "rubric_count": rubric_count},
                force=final_parse_status != "full",
            )
            if final_parse_status == "full":
                break
        except Exception as exc:  # noqa: BLE001
            last_exception = repr(exc)
            last_debug_paths = _write_scored_debug_file(
                tag="scored_multi_exception",
                sample_key=sample_key,
                attempt_idx=attempt_idx,
                grader_prompt=grader_prompt,
                raw_response=last_raw_response,
                parse_status="exception",
                missing_count=group_size * rubric_count,
                expected_count=group_size * rubric_count,
                pi_mode=_effective_pi_mode(pi_mode),
                response_metadata=last_response_metadata,
                exception=exc,
                extra={"data_source": data_source, "group_size": group_size, "rubric_count": rubric_count},
                force=True,
            )
            if attempt_idx == 2:
                raise

    _print_parse_failure_context(
        tag="scored_multi",
        sample_key=sample_key,
        parse_status=final_parse_status,
        expected_count=group_size * rubric_count,
        missing_count=final_missing_count,
        raw_response=last_raw_response,
        grader_prompt=grader_prompt,
        response_metadata=last_response_metadata,
        debug_paths=last_debug_paths,
    )
    _maybe_raise_or_warn_parse_failure(
        tag="scored_multi",
        sample_key=sample_key,
        parse_status=final_parse_status,
        expected_count=group_size * rubric_count,
        missing_count=final_missing_count,
        debug_paths=last_debug_paths,
    )

    debug_fields = _rollout_debug_fields(
        raw_response=last_raw_response,
        grader_prompt=grader_prompt,
        response_metadata=last_response_metadata,
        debug_paths=last_debug_paths,
        exception=last_exception,
    )

    results = []
    stats = []
    for response_local_idx, sample_idx in enumerate(sample_indices, start=1):
        response_results = grouped_results.get(response_local_idx, {})
        for rubric_local_idx, rubric_idx in enumerate(task["rubric_indices"], start=1):
            results.append(
                {
                    "sample_idx": sample_idx,
                    "rubric_idx": rubric_idx,
                    "result": response_results.get(
                        rubric_local_idx,
                        _default_missing_scored_result(rubric_items[rubric_local_idx - 1], data_source),
                    ),
                }
            )
        judged_count = len(response_results)
        parse_status = "failed"
        if judged_count:
            parse_status = "partial" if judged_count < rubric_count else "full"
        stats.append(
            {
                "sample_idx": sample_idx,
                "num_llm_rubrics": rubric_count,
                "num_llm_judged": judged_count,
                "num_llm_missing_judgments": max(0, rubric_count - judged_count),
                "llm_parse_status": parse_status if final_parse_status != "failed" else "failed",
                "llm_retries_used": retries_used,
                "llm_group_size": group_size,
                "llm_request_share": 1.0 / group_size,
                **debug_fields,
            }
        )

    return {"results": results, "stats": stats}


def _llm_group_key(
    data_source: str,
    prompt: List[Dict[str, str]],
    rubric_items: List[RubricItem],
    rubric_indices: List[int],
) -> str:
    return _canonical_json(
        {
            "data_source": data_source,
            "prompt": prompt,
            "rubric_indices": rubric_indices,
            "rubrics": [rubric_item.to_dict() for rubric_item in rubric_items],
        }
    )


def _build_tasks(
    batch_data: List[Tuple[str, str, str, Dict[str, Any]]],
    *,
    met_threshold: float,
    pi_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rule_tasks: List[Dict[str, Any]] = []
    llm_tasks: List[Dict[str, Any]] = []
    active_llm_groups: Dict[str, Dict[str, Any]] = {}

    def new_llm_group(
        *,
        group_key: str,
        data_source: str,
        prompt: List[Dict[str, str]],
        rubric_items: List[RubricItem],
        rubric_indices: List[int],
    ) -> Dict[str, Any]:
        group = {
            "sample_indices": [],
            "data_source": data_source,
            "prompt": prompt,
            "responses": [],
            "rubric_items": rubric_items,
            "rubric_indices": rubric_indices,
        }
        active_llm_groups[group_key] = group
        llm_tasks.append(group)
        return group

    for sample_idx, (data_source, solution_str, _, extra_info) in enumerate(batch_data):
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
                        "data_source": data_source,
                        "response": solution_str,
                        "rubric_item": rubric_item,
                        "met_threshold": met_threshold,
                        "pi_mode": pi_mode,
                    }
                )
            else:
                llm_indices.append(rubric_idx)

        if llm_indices:
            max_responses_per_judge = max(1, GEAR_MAX_RESPONSES_PER_JUDGE)
            max_rubrics_per_judge = max(1, GEAR_MAX_RUBRICS_PER_JUDGE)

            for start in range(0, len(llm_indices), max_rubrics_per_judge):
                chunk_indices = llm_indices[start : start + max_rubrics_per_judge]
                chunk_rubric_items = [rubric_items[idx] for idx in chunk_indices]

                if GEAR_GROUP_RESPONSES_PER_PROMPT:
                    group_key = _llm_group_key(data_source, prompt, chunk_rubric_items, chunk_indices)
                    llm_group = active_llm_groups.get(group_key)
                    if llm_group is None or len(llm_group["sample_indices"]) >= max_responses_per_judge:
                        llm_group = new_llm_group(
                            group_key=group_key,
                            data_source=data_source,
                            prompt=prompt,
                            rubric_items=chunk_rubric_items,
                            rubric_indices=chunk_indices,
                        )
                else:
                    llm_group = new_llm_group(
                        group_key=f"sample:{sample_idx}:rubrics:{start}",
                        data_source=data_source,
                        prompt=prompt,
                        rubric_items=chunk_rubric_items,
                        rubric_indices=chunk_indices,
                    )

                llm_group["sample_indices"].append(sample_idx)
                llm_group["responses"].append(solution_str)

    return rule_tasks, llm_tasks


def _standard_scored_score(rubric_items: List[RubricItem], grading_results: List[Dict[str, Any]]) -> float:
    total_possible_points = sum(item.points for item in rubric_items if item.points > 0)
    if total_possible_points <= 0:
        return 0.0
    achieved_points = sum(
        item.points * float(grading_result.get("normalized_score", 0.0))
        for item, grading_result in zip(rubric_items, grading_results)
        if item.points > 0
    )
    return achieved_points / total_possible_points


def _effective_final_reward_mode(final_reward_mode: str) -> str:
    normalized = str(final_reward_mode or "aggregate").strip().lower()
    if normalized in {"standard", "weighted", "baseline", "points"}:
        return "standard"
    return "aggregate"


def _empty_result(
    *,
    ground_truth: str,
    aggregation_mode: str,
    sample_llm_stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
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
        "flat_q_list": [],
        "hard_q_list": [],
        "dag_q_list": [],
        "node_types": [],
        "graph_edges": [],
        "criteria_met_list": [],
        "rubric_ids": [],
        "normalized_score_list": [],
        "awarded_score_list": [],
        "max_score_list": [],
        "score_type_list": [],
        "subitem_results_list": [],
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
        "num_llm_rubrics": sample_llm_stats["num_llm_rubrics"],
        "num_llm_judged": sample_llm_stats["num_llm_judged"],
        "num_llm_missing_judgments": sample_llm_stats["num_llm_missing_judgments"],
        "llm_parse_status": sample_llm_stats["llm_parse_status"],
        "llm_retries_used": sample_llm_stats["llm_retries_used"],
        "llm_group_size": sample_llm_stats["llm_group_size"],
        "llm_request_share": sample_llm_stats["llm_request_share"],
    }


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    max_workers_per_url: int = MAX_CONCURRENT_WORKERS,
    aggregation_mode: str = "dag",
    normalization_mode: str = "positive_sum",
    inference_mode: str = "approx",
    exact_if_num_nodes_le: int = 10,
    lambda_by_edge_type: Optional[Dict[str, float]] = None,
    graph_source: str = "dataset",
    acc_mode: str = "standard",
    score_met_threshold: float = 0.7,
    pi_mode: str = "judge_prob",
    final_reward_mode: str = "aggregate",
    **_: Any,
) -> List[Dict[str, Any]]:
    del graph_source

    met_threshold = _clamp_probability(score_met_threshold, default=0.7)
    effective_pi_mode = _effective_pi_mode(pi_mode)
    effective_final_reward_mode = _effective_final_reward_mode(final_reward_mode)
    batch_data = list(zip(data_sources, solution_strs, ground_truths, extra_infos))
    rule_tasks, llm_tasks = _build_tasks(batch_data, met_threshold=met_threshold, pi_mode=effective_pi_mode)

    sample_results: Dict[int, Dict[int, Dict[str, Any]]] = {}
    llm_task_stats: Dict[int, Dict[str, Any]] = {}
    for task in rule_tasks:
        result = _process_rule_task(task)
        sample_results.setdefault(result["sample_idx"], {})[result["rubric_idx"]] = result["result"]

    if llm_tasks:
        grader = get_global_grader()
        total_workers = max(1, min(len(llm_tasks), max_workers_per_url * max(1, len(grader.base_urls))))
        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            futures = [executor.submit(_process_llm_task, task, grader, met_threshold, effective_pi_mode) for task in llm_tasks]
            for future in as_completed(futures):
                llm_task_output = future.result()
                for result in llm_task_output["results"]:
                    sample_results.setdefault(result["sample_idx"], {})[result["rubric_idx"]] = result["result"]
                for stats in llm_task_output["stats"]:
                    llm_task_stats[stats["sample_idx"]] = stats

    results: List[Dict[str, Any]] = []
    for sample_idx, (data_source, _, ground_truth, extra_info) in enumerate(batch_data):
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
        for rubric_idx, rubric_item in enumerate(rubric_items):
            grading_results.append(
                sample_results.get(sample_idx, {}).get(
                    rubric_idx,
                    _default_missing_scored_result(rubric_item, data_source),
                )
            )

        if not prompt or not rubric_items:
            empty_result = _empty_result(
                ground_truth=ground_truth,
                aggregation_mode=aggregation_mode,
                sample_llm_stats=sample_llm_stats,
            )
            if _debug_enabled():
                empty_result.update({key: value for key, value in sample_llm_stats.items() if key.startswith("llm_")})
            results.append(empty_result)
            continue

        standard_score = _standard_scored_score(rubric_items, grading_results)
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

        selected_reward = standard_score if effective_final_reward_mode == "standard" else aggregate.reward
        selected_acc = selected_reward > 0.5
        reported_acc = (
            standard_score > 0.5
            if effective_final_reward_mode == "standard"
            else _select_reported_acc(acc_mode=acc_mode, standard_score=standard_score, aggregate=aggregate)
        )

        result_item = {
            "score": selected_reward,
            "acc": reported_acc,
            "ground_truth": ground_truth,
            "aggregation_mode": aggregate.aggregation_mode,
            "baseline_reward": standard_score,
            "baseline_acc": standard_score > 0.5,
            "standard_score": standard_score,
            "standard_acc": standard_score > 0.5,
            "selected_reward": selected_reward,
            "selected_acc": selected_acc,
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
            "normalized_score_list": [item["normalized_score"] for item in grading_results],
            "awarded_score_list": [item["awarded_score"] for item in grading_results],
            "max_score_list": [item["max_score"] for item in grading_results],
            "score_type_list": [item["score_type"] for item in grading_results],
            "subitem_results_list": [item.get("subitem_results", []) for item in grading_results],
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
            "num_llm_rubrics": sample_llm_stats["num_llm_rubrics"],
            "num_llm_judged": sample_llm_stats["num_llm_judged"],
            "num_llm_missing_judgments": sample_llm_stats["num_llm_missing_judgments"],
            "llm_parse_status": sample_llm_stats["llm_parse_status"],
            "llm_retries_used": sample_llm_stats["llm_retries_used"],
            "llm_group_size": sample_llm_stats["llm_group_size"],
            "llm_request_share": sample_llm_stats["llm_request_share"],
        }

        if _debug_enabled():
            result_item.update({key: value for key, value in sample_llm_stats.items() if key.startswith("llm_")})

        results.append(result_item)

    if results and _should_print_debug_summary():
        num_graph_edges_vals = [float(item.get("num_graph_edges", 0.0)) for item in results]
        has_graph_edges_vals = [float(item.get("has_graph_edges", 0.0)) for item in results]
        llm_missing_vals = [float(item.get("num_llm_missing_judgments", 0.0)) for item in results]
        llm_parse_statuses = [str(item.get("llm_parse_status", "not_applicable")) for item in results]
        print(
            "[SCORED_GEAR_DEBUG] "
            f"mode={aggregation_mode} pi_mode={effective_pi_mode} final_reward_mode={effective_final_reward_mode} "
            f"batch_size={len(results)} "
            f"graph_nonempty_ratio={_mean(has_graph_edges_vals):.3f} "
            f"avg_edges={_mean(num_graph_edges_vals):.3f} "
            f"avg_llm_missing={_mean(llm_missing_vals):.3f} "
            f"llm_parse_full_ratio={_mean([1.0 if status == 'full' else 0.0 for status in llm_parse_statuses]):.3f} "
            f"avg_standard_score={_mean([float(item.get('standard_score', 0.0)) for item in results]):.3f} "
            f"avg_selected_score={_mean([float(item.get('score', 0.0)) for item in results]):.3f} "
            f"VLLM_BASE_URL={os.getenv('VLLM_BASE_URL')} "
            f"GEAR_MAX_RUBRICS_PER_JUDGE={GEAR_MAX_RUBRICS_PER_JUDGE} "
            f"GEAR_RAISE_PARSE_FAILURE={int(GEAR_RAISE_PARSE_FAILURE)}"
        )

    return results