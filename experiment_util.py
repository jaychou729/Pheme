#!/usr/bin/env python3
"""
Utility functions for the PHEME four behavioral-signature experiments.

This module contains:
- Bailian/DashScope request and stance parsing helpers
- PHEME thread loading and graph construction
- controlled and real-neighbor prompt builders
- controlled and real-graph experiment runners
- condition and pairwise analysis
- agent selection and graph summary helpers

The experiment workflow itself is in ``pheme_four_effects.ipynb``.
Set the Bailian/DashScope key through the environment instead of hardcoding it:

    export DASHSCOPE_API_KEY="your_dashscope_key"
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# ============================================================
# Runtime globals
# ============================================================

API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.getenv("DASHSCOPE_API_KEY", os.getenv("API_KEY", ""))
MODEL = "deepseek-v4-pro"
INIT_MODEL = "deepseek-v4-pro"

MAX_WORKERS = 5
SLEEP_BETWEEN = 0.2
MAX_NEIGHBORS_IN_PROMPT = 8
THINKING_CONTROL_MODELS = {
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v3.2",
    "deepseek-v3.2-exp",
    "deepseek-v3.1",
}

# 1=deny rumor, 5=support rumor
OPINION_LABELS = {
    1: "Strongly Deny",
    2: "Deny",
    3: "Question or Neutral",
    4: "Support",
    5: "Strongly Support",
}

STANCE_SCORE_GUIDE = """1 = strongly denies, rejects, or opposes the source post
2 = denies, rejects, opposes, or disagrees with the source post
3 = questions, doubts, neutral, unclear, uncertain, unrelated, or only asks for information
4 = supports, accepts, or agrees with the source post
5 = strongly supports or explicitly confirms the source post"""

GRAPH: Dict[int, List[int]] = {}
AGENT_NAMES: Dict[int, str] = {}
AGENT_TEXTS: Dict[int, str] = {}
INITIAL_OPINIONS: Dict[int, int] = {}
TOPIC: str = ""
CURRENT_THREAD_ID: str = ""

call_count = 0

SYSTEM_MSG = (
    "You are a strict stance classification model for social media discussions. "
    "Use the scoring rubric and output format specified in the user prompt. "
    "Do not add explanations unless the prompt explicitly asks for them."
)


# ============================================================
# Notebook-facing runtime configuration
# ============================================================

# configure_runtime 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def configure_runtime(
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    init_model: Optional[str] = None,
    max_workers: Optional[int] = None,
    sleep_between: Optional[float] = None,
    max_neighbors_in_prompt: Optional[int] = None,
    reset_call_count: bool = True,
) -> None:
    """Configure module-level runtime state from a notebook or another script."""
    global API_URL, API_KEY, MODEL, INIT_MODEL
    global MAX_WORKERS, SLEEP_BETWEEN, MAX_NEIGHBORS_IN_PROMPT, call_count

    if api_url is not None:
        API_URL = str(api_url)
    if api_key is not None:
        API_KEY = str(api_key)
    if model is not None:
        MODEL = str(model)
    if init_model is not None:
        INIT_MODEL = str(init_model)
    if max_workers is not None:
        MAX_WORKERS = max(1, int(max_workers))
    if sleep_between is not None:
        SLEEP_BETWEEN = max(0.0, float(sleep_between))
    if max_neighbors_in_prompt is not None:
        MAX_NEIGHBORS_IN_PROMPT = max(1, int(max_neighbors_in_prompt))
    if reset_call_count:
        call_count = 0


# set_experiment_data 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def set_experiment_data(
    graph: Dict[int, List[int]],
    agent_names: Dict[int, str],
    agent_texts: Dict[int, str],
    initial_opinions: Dict[int, int],
    topic: str,
) -> None:
    """Store one loaded PHEME thread in the module runtime state."""
    global GRAPH, AGENT_NAMES, AGENT_TEXTS, INITIAL_OPINIONS, TOPIC

    GRAPH = graph
    AGENT_NAMES = agent_names
    AGENT_TEXTS = agent_texts
    INITIAL_OPINIONS = initial_opinions
    TOPIC = topic


# _safe_stance_filename_part 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def _safe_stance_filename_part(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


# build_stance_file_path 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_stance_file_path(
    stance_dir: Path,
    *,
    selected_thread: Path,
    init_model: str,
    use_llm_init: bool,
) -> Path:
    """Build the canonical stance-cache path from thread id and model only."""
    model_key = init_model if use_llm_init else "heuristic"
    thread_id = _safe_stance_filename_part(Path(selected_thread).name)
    model_name = _safe_stance_filename_part(model_key)
    filename = f"{thread_id}_{model_name}.json"
    return Path(stance_dir) / filename


# find_stance_file_path 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def find_stance_file_path(
    stance_dir: Path,
    *,
    selected_thread: Path,
    init_model: str,
    use_llm_init: bool,
) -> Optional[Path]:
    """Find a stance file by thread id and model, including legacy names."""
    stance_dir = Path(stance_dir)
    canonical_path = build_stance_file_path(
        stance_dir,
        selected_thread=selected_thread,
        init_model=init_model,
        use_llm_init=use_llm_init,
    )

    if canonical_path.exists():
        return canonical_path

    model_key = init_model if use_llm_init else "heuristic"
    thread_id = _safe_stance_filename_part(Path(selected_thread).name)
    model_name = _safe_stance_filename_part(model_key)
    legacy_candidates = sorted(
        stance_dir.glob(f"{thread_id}_*{model_name}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    return legacy_candidates[0] if legacy_candidates else None


# save_initial_opinions 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def save_initial_opinions(
    output_path: Path,
    *,
    selected_thread: Path,
    topic: str,
    agent_names: Dict[int, str],
    agent_texts: Dict[int, str],
    initial_opinions: Dict[int, int],
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save initialized stance scores for deterministic reuse."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "selected_thread": str(selected_thread),
        "topic": topic,
        "config": config or {},
        "initial_opinions": {
            str(agent_id): int(score)
            for agent_id, score in sorted(initial_opinions.items())
        },
        "agent_names": {
            str(agent_id): name
            for agent_id, name in sorted(agent_names.items())
        },
        "agent_texts": {
            str(agent_id): text
            for agent_id, text in sorted(agent_texts.items())
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Stance Cache] Saved initial stances: {output_path}")
    return output_path


# load_initial_opinions 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def load_initial_opinions(
    input_path: Path,
) -> Dict[int, int]:
    """Load initialized stance scores from a stance-cache JSON file."""
    input_path = Path(input_path)

    with input_path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)

    raw_opinions = payload.get("initial_opinions", payload)
    opinions = {
        int(agent_id): int(score)
        for agent_id, score in raw_opinions.items()
    }

    print(f"[Stance Cache] Loaded initial stances: {input_path}")
    return opinions


# load_and_set_pheme_thread 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def load_and_set_pheme_thread(
    event_dir: Path,
    *,
    thread_id: Optional[str] = None,
    min_reactions: int = 10,
    max_agents: int = 200,
    use_llm_init: bool = True,
    stance_dir: Optional[Path] = None,
    force_recompute_stance: bool = False,
) -> Tuple[Path, Dict[int, List[int]], Dict[int, str], Dict[int, str], Dict[int, int], str]:
    """Select, load, register, and optionally cache PHEME initial stances."""
    global CURRENT_THREAD_ID

    selected_thread = select_pheme_thread(
        Path(event_dir),
        min_reactions=min_reactions,
        thread_id=thread_id,
    )
    CURRENT_THREAD_ID = selected_thread.name

    stance_path: Optional[Path] = None
    cached_opinions: Optional[Dict[int, int]] = None

    if stance_dir is not None:
        canonical_stance_path = build_stance_file_path(
            Path(stance_dir),
            selected_thread=selected_thread,
            init_model=INIT_MODEL,
            use_llm_init=use_llm_init,
        )
        matched_stance_path = find_stance_file_path(
            Path(stance_dir),
            selected_thread=selected_thread,
            init_model=INIT_MODEL,
            use_llm_init=use_llm_init,
        )
        stance_path = canonical_stance_path

        if matched_stance_path is not None and not force_recompute_stance:
            cached_opinions = load_initial_opinions(
                matched_stance_path,
            )
            stance_path = matched_stance_path

    graph, names, texts, opinions, topic = load_pheme_thread(
        selected_thread,
        max_agents=max_agents,
        use_llm_init=use_llm_init,
        initial_opinions_override=cached_opinions,
    )

    if stance_dir is not None and (
        force_recompute_stance or cached_opinions is None
    ):
        stance_path = build_stance_file_path(
            Path(stance_dir),
            selected_thread=selected_thread,
            init_model=INIT_MODEL,
            use_llm_init=use_llm_init,
        )
        save_initial_opinions(
            stance_path,
            selected_thread=selected_thread,
            topic=topic,
            agent_names=names,
            agent_texts=texts,
            initial_opinions=opinions,
            config={
                "max_agents": max_agents,
                "use_llm_init": use_llm_init,
                "init_model": INIT_MODEL if use_llm_init else "heuristic",
            },
        )

    set_experiment_data(graph, names, texts, opinions, topic)
    return selected_thread, graph, names, texts, opinions, topic


# save_experiment_bundle 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def save_experiment_bundle(
    output_path: Path,
    *,
    config: Dict[str, Any],
    selected_thread: Path,
    topic: str,
    graph_summary: Dict[str, Any],
    analyses: Dict[str, Any],
    pairwise: Dict[str, Any],
    results: Dict[str, Any],
    timestamp: Optional[str] = None,
) -> Path:
    """Save the complete notebook experiment output as JSON."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "config": config,
        "selected_thread": str(selected_thread),
        "topic": topic,
        "graph_summary": graph_summary,
        "analyses": analyses,
        "pairwise": pairwise,
        "results": results,
        "total_api_calls": call_count,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


# build_llm_output_file_path 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_llm_output_file_path(
    output_dir: Path,
    *,
    selected_thread: Path,
    model: str,
    condition: Optional[str] = None,
) -> Path:
    """Build a comment/score output path using the stance-cache naming style."""
    thread_id = _safe_stance_filename_part(Path(selected_thread).name)
    model_name = _safe_stance_filename_part(model)
    parts = [thread_id, model_name]

    if condition:
        parts.append(
            _safe_stance_filename_part(condition)
        )

    return Path(output_dir) / f"{'_'.join(parts)}.json"


# save_llm_comment_score_outputs 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def save_llm_comment_score_outputs(
    output_path: Path,
    *,
    selected_thread: Path,
    topic: str,
    model: str,
    results: Sequence[Dict[str, Any]],
    condition: Optional[str] = None,
    init_model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """Save generated natural-language comments and classified scores."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for result in sorted(
        results,
        key=lambda row: (
            row.get("agent", -1),
            row.get("rep", -1),
        ),
    ):
        records.append({
            "condition": result.get("condition", condition),
            "agent": result.get("agent"),
            "name": result.get("name"),
            "rep": result.get("rep"),
            "old_score": result.get("old_score"),
            "new_score": result.get("new_score"),
            "classification_score": result.get("new_score"),
            "change": result.get("change"),
            "generated_comment": result.get("generated_comment"),
            "classification_response": result.get(
                "classification_response",
                result.get("raw_response"),
            ),
            "generation_prompt": result.get("generation_prompt"),
            "classification_prompt": result.get("classification_prompt"),
            "neighbor_count": result.get("neighbor_count"),
            "opposite_neighbor_count": result.get(
                "opposite_neighbor_count"
            ),
            "support_neighbor_count": result.get(
                "support_neighbor_count"
            ),
            "neighbor_ids": result.get("neighbor_ids"),
            "neighbor_sources": result.get("neighbor_sources"),
            "neighbor_relations": result.get("neighbor_relations"),
            "real_neighbor_count": result.get("real_neighbor_count"),
            "random_neighbor_count": result.get("random_neighbor_count"),
            "generation_tokens": result.get("generation_tokens"),
            "classification_tokens": result.get("classification_tokens"),
            "tokens": result.get("tokens"),
        })

    payload = {
        "timestamp": timestamp,
        "selected_thread": str(selected_thread),
        "thread_id": Path(selected_thread).name,
        "topic": topic,
        "model": model,
        "init_model": init_model,
        "condition": condition,
        "config": config or {},
        "records": records,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[LLM Output] Saved comments and scores: {output_path}")
    return output_path


# ============================================================
# Bailian/DashScope calls
# ============================================================

# llm_call 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def llm_call(
    system_msg: str,
    user_msg: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
    max_retries: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """Call Bailian/DashScope OpenAI-compatible chat completion API."""
    global call_count
    call_count += 1

    if not API_KEY:
        raise RuntimeError(
            "API key is empty. Please run: "
            "export DASHSCOPE_API_KEY='your_dashscope_key' or pass --api-key."
        )

    selected_model = model or MODEL

    system_msg = str(system_msg or "").strip()
    user_msg = str(user_msg or "").strip()

    if not system_msg:
        raise ValueError("system_msg is empty")

    if not user_msg:
        raise ValueError("user_msg is empty")

    # GPT-5 Mini 最低要求为 16
    max_tokens = max(16, int(max_tokens))

    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    if selected_model in THINKING_CONTROL_MODELS:
        payload["enable_thinking"] = False

    # GPT-5 Mini 不要发送 temperature=0.0。
    # 其他模型仍然可以使用调用方传入的 temperature。
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                try:
                    error_body = resp.json()
                    error_text = json.dumps(
                        error_body,
                        indent=2,
                        ensure_ascii=False,
                    )
                except ValueError:
                    error_text = resp.text

                print(f"  [API Error] HTTP {resp.status_code}")
                print(f"  [Model] {selected_model}")
                print(f"  [Response] {error_text}")

                # 400 是请求参数错误，重复发送同样的请求没有意义
                if resp.status_code == 400:
                    return "ERROR", {}

                # 只有限流或服务器临时错误才重试
                if resp.status_code not in {
                    408, 429, 500, 502, 503, 504
                }:
                    return "ERROR", {}

                last_error = RuntimeError(
                    f"HTTP {resp.status_code}: {error_text}"
                )

            else:
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    print("  [API Error] Response has no choices:")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    return "ERROR", {}

                message = choices[0].get("message", {})
                content = message.get("content")

                if not isinstance(content, str) or not content.strip():
                    print("  [API Error] Model returned empty content:")
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    return "ERROR", data.get("usage", {}) or {}

                usage = data.get("usage", {}) or {}
                return content.strip(), usage

        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"  [Retry {attempt}] API timeout: {e}")

        except requests.exceptions.ConnectionError as e:
            last_error = e
            print(f"  [Retry {attempt}] Connection error: {e}")

        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  [Retry {attempt}] Request error: {e}")

        except (KeyError, IndexError, ValueError, TypeError) as e:
            print(f"  [Failed] Response parsing error: {e}")
            return "ERROR", {}

        if attempt < max_retries:
            time.sleep(1.5 * attempt)

    print(f"  [Failed] API call failed after retries: {last_error}")
    return "ERROR", {}


# parse_opinion 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def parse_opinion(text: str) -> Optional[int]:
    """Extract a stance score from model output."""
    if not text:
        return None

    text = text.strip()

    # Prefer the expected formats: "4", "4 - Support", or "Score: 4".
    match = re.match(
        r"^\s*(?:score\s*[:=-]\s*)?([1-5])\s*(?:$|[-:]\s*(?:strongly\s+)?(?:support|deny|neutral|question|oppose)\b.*)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1))

    # If the model returned a short sentence but only one stance number appears,
    # accept it. If multiple numbers appear, fail instead of silently taking the
    # wrong one from an explanation.
    matches = re.findall(r"\b[1-5]\b", text)
    if len(matches) == 1:
        return int(matches[0])
    if len(matches) > 1:
        return None

    lower = re.sub(r"\s+", " ", text.lower()).strip(" .:-")
    if lower == "strongly support":
        return 5
    if lower in {"strongly deny", "strongly oppose"}:
        return 1
    if lower == "support":
        return 4
    if lower in {"deny", "oppose"}:
        return 2
    if lower in {"question", "neutral", "uncertain"}:
        return 3

    return None


# ============================================================
# PHEME loading
# ============================================================

# load_json 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# get_tweet_id 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_tweet_id(tweet: Dict[str, Any], fallback: str) -> str:
    return str(tweet.get("id_str") or tweet.get("id") or fallback)


# get_tweet_text 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_tweet_text(tweet: Dict[str, Any]) -> str:
    text = tweet.get("full_text") or tweet.get("text") or tweet.get("body") or ""
    return str(text).replace("\n", " ").strip()


# get_screen_name 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_screen_name(tweet: Dict[str, Any], tweet_id: str) -> str:
    user = tweet.get("user", {}) or {}
    name = user.get("screen_name") or user.get("name")
    if name:
        return str(name).replace("\n", " ").strip()
    return f"user_{tweet_id[-5:]}"

# get_created_at 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_created_at(tweet: Dict[str, Any]):
    raw = tweet.get("created_at")
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


# flatten_structure_tree 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def flatten_structure_tree(tree: Dict[str, Any], parent: Optional[str] = None) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    if not isinstance(tree, dict):
        return edges
    for node, children in tree.items():
        node = str(node)
        if parent is not None:
            edges.append((str(parent), node))
        if isinstance(children, dict):
            edges.extend(flatten_structure_tree(children, node))
    return edges


# get_reply_parent_id 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_reply_parent_id(tweet: Dict[str, Any]) -> Optional[str]:
    parent = tweet.get("in_reply_to_status_id_str") or tweet.get("in_reply_to_status_id")
    return str(parent) if parent is not None else None


# build_edges_from_reply_metadata 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_edges_from_reply_metadata(
    tweets: Dict[str, Dict[str, Any]],
    source_id: str,
) -> List[Tuple[str, str]]:
    edges = []
    for tid, tweet in tweets.items():
        if tid == source_id:
            continue
        parent_id = get_reply_parent_id(tweet)
        if parent_id and parent_id in tweets:
            edges.append((parent_id, tid))
        else:
            edges.append((source_id, tid))
    return edges


# heuristic_initial_opinion 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def heuristic_initial_opinion(text: str, is_source: bool = False) -> int:
    """Fallback stance initializer."""
    if is_source:
        return 4

    lower = text.lower()
    strong_deny_words = ["fake", "hoax", "completely false", "totally false", "definitely false"]
    deny_words = [
        "false", "not true", "incorrect", "debunk", "rumor", "rumour", "no evidence",
        "unconfirmed", "didn't", "did not", "never happened", "not confirmed",
        "can't confirm", "cannot confirm",
    ]
    strong_support_words = ["confirmed", "definitely true", "absolutely true"]
    support_words = ["true", "agree", "yes", "exactly", "right", "correct", "this is true"]
    question_words = ["?", "source", "really", "is this true", "confirmed?", "any proof", "evidence"]

    if any(w in lower for w in strong_deny_words):
        return 1
    if any(w in lower for w in strong_support_words):
        return 5
    if any(w in lower for w in deny_words):
        return 2
    if any(w in lower for w in support_words):
        return 4
    if any(w in lower for w in question_words):
        return 3
    return 3


# build_stance_classification_prompt 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def _stance_prompt_topic_key(source_text: str) -> str:
    """Choose the stance rubric from the loaded thread id or source text."""
    thread_id = str(CURRENT_THREAD_ID or "").strip()
    source_lower = str(source_text or "").lower()

    if thread_id == "524947716393414656" or (
        "issues with gun control" in source_lower
        and "ottawa shooting" in source_lower
    ):
        return "gun_control"

    if thread_id == "524934142958788608" or (
        "flurry of muslim hate tweets" in source_lower
        and "we are better than this" in source_lower
    ):
        return "anti_muslim_hate"

    if thread_id == "525046443103354880" or (
        "muslim convert michael zehaf-bibeau" in source_lower
        or "identifies muslim convert" in source_lower
    ):
        return "muslim_convert_framing"

    return "generic"


def _stance_prompt_spec(source_text: str) -> Dict[str, str]:
    """Return the topic-specific stance definitions used inside the prompt."""
    topic_key = _stance_prompt_topic_key(source_text)

    if topic_key == "gun_control":
        return {
            "discussion": (
                "the gun-control / gun-violence debate raised by the Ottawa shooting"
            ),
            "important": """- The source post argues that America has a gun-control / gun-violence problem
  and that the tweet is criticizing America's gun-control issue, not
  minimizing the Ottawa shooting.
- Score 1 (Strongly Deny): The comment STRONGLY argues AGAINST stricter gun
  control, or STRONGLY insists guns are NOT the problem. Examples: defending
  gun rights, arguing criminals not guns are the issue, citing self-defense
  needs, or saying gun control does not work.
- Score 2 (Deny): The comment argues AGAINST stricter gun control or says guns
  are not the problem, but less forcefully than score 1.
- Score 3 (Neutral/Question): The comment is neutral, unclear, asks a question,
  discusses the shooting without taking a clear position on gun control, only
  jokes/socializes, only shares a link, or discusses free speech rather than gun
  policy.
- Score 4 (Support): The comment argues FOR stricter gun control, says America
  HAS a gun problem, defends the source post's gun-control framing, or argues
  that regulation can work.
- Score 5 (Strongly Support): The comment STRONGLY argues FOR stricter gun
  control, or STRONGLY insists America HAS a gun-violence problem. Examples:
  calling for gun reform, citing gun-death statistics to support gun control,
  or comparing the US unfavorably to countries with stricter laws.""",
            "critical": """- Merely MENTIONING guns, shootings, or violence does NOT mean the comment
  supports stricter gun control. You must identify the POLICY POSITION, not
  the topic being discussed.
- A comment that says "this is about mental health, not guns" is DENY (1-2),
  not support, even though it discusses the shooting.
- A comment that says "we need better background checks", supports limiting
  guns, magazines, or ammunition, or says regulation can work is SUPPORT (4-5).
- A comment that says "gun control won't stop criminals", "guns don't kill
  people", "good guys with guns", or defends broad gun rights is DENY (1-2).
- A comment that merely expresses sadness about the shooting, discusses free
  speech, jokes, socializes, asks for sources, or only posts a link without a
  clear policy position is NEUTRAL (3).
- If the comment's position on gun control is unclear, classify as 3.""",
            "support_rule": (
                "the comment clearly supports the source post's gun-control / "
                "gun-violence framing."
            ),
        }

    if topic_key == "anti_muslim_hate":
        return {
            "discussion": "Muslim hate / Islamophobia after the Ottawa shooting",
            "important": """- The source post argues that, with limited facts about the Ottawa shooting,
  people should not spread Muslim hate, Islamophobia, or blame Muslims/Islam as
  a group. It says people are better than generalized anti-Muslim hate.
- Score 1 (Strongly Deny): The comment STRONGLY rejects the source post by
  blaming Muslims/Islam as a group, endorsing anti-Muslim hate, or insisting
  that Muslims/Islam are the problem.
- Score 2 (Deny): The comment rejects the source post less forcefully, for
  example by defending suspicion toward Muslims, dismissing concern about
  Islamophobia, or arguing that Muslim identity should be emphasized as the
  main issue.
- Score 3 (Neutral/Question): The comment is neutral, unclear, asks a question,
  only reports facts, discusses the shooting without judging anti-Muslim hate,
  or seeks common ground without clearly agreeing or disagreeing.
- Score 4 (Support): The comment supports the source post by opposing
  anti-Muslim hate, warning against generalizing from one attacker to all
  Muslims, or saying people should wait for facts.
- Score 5 (Strongly Support): The comment STRONGLY supports the source post by
  explicitly condemning Islamophobia, strongly defending Muslims as a group, or
  forcefully rejecting collective blame after the Ottawa shooting.""",
            "critical": """- Merely MENTIONING Muslims, Islam, terrorism, shootings, or violence does NOT
  determine the stance. You must identify the comment's position toward
  generalized Muslim hate / Islamophobia.
- A comment that says "not all Muslims", "stop blaming Muslims", "wait for
  facts", "do not spread hate", or condemns Islamophobia is SUPPORT (4-5).
- A comment that blames Muslims/Islam as a group, endorses suspicion toward
  Muslims generally, or dismisses concern about anti-Muslim hate is DENY (1-2).
- A comment that only reports the suspect's identity, asks for information,
  expresses sadness, or discusses the attack without judging anti-Muslim hate
  is NEUTRAL (3).
- If the comment's position on Muslim hate / Islamophobia is unclear, classify
  as 3.""",
            "support_rule": (
                "the comment clearly supports the source post's anti-hate / "
                "anti-generalization claim."
            ),
        }

    if topic_key == "muslim_convert_framing":
        return {
            "discussion": (
                "whether the Ottawa shooter's Muslim-convert identity is "
                "relevant to report"
            ),
            "important": """- The source post reports the Ottawa shooter as a Muslim convert and frames
  that identity as a relevant fact in the attack report.
- Score 1 (Strongly Deny): The comment STRONGLY rejects the source post's
  framing by saying that mentioning Muslim/Islam is biased, Islamophobic,
  stereotyping, fear-mongering, or completely irrelevant to the attack.
- Score 2 (Deny): The comment rejects the framing less forcefully, for example
  by questioning why the Muslim-convert identity is mentioned, comparing it to
  unreported Christian identities, or arguing that the report unfairly singles
  out Islam/Muslims.
- Score 3 (Neutral/Question): The comment is neutral, unclear, asks a question,
  only repeats or reports facts, discusses the shooting without judging whether
  Muslim-convert identity is relevant, or seeks common ground without clearly
  agreeing or disagreeing.
- Score 4 (Support): The comment supports the source post's framing by saying
  the Muslim-convert identity is a relevant fact, may help explain motive, or is
  legitimate context for the Ottawa attack.
- Score 5 (Strongly Support): The comment STRONGLY supports the framing by
  explicitly arguing that Islam/Islamism/Muslim extremism is central to the
  attack, that the media should identify it, or that avoiding the label hides
  the truth.""",
            "critical": """- Merely MENTIONING Muslims, Islam, terrorism, shootings, or violence does NOT
  determine the stance. You must identify whether the comment accepts or
  rejects the relevance of mentioning the shooter's Muslim-convert identity.
- A comment that says "it is relevant", "it suggests motive", "facts are
  facts", "why hide it", or links the attack to Islamist/Muslim extremism is
  SUPPORT (4-5).
- A comment that says "why mention Muslim", "this is stereotyping", "this is
  bias", "not all Muslims", "Islam is not responsible", or compares it to not
  labeling Christian shooters is DENY (1-2).
- A comment can reject anti-Muslim generalization but still be NEUTRAL (3) if
  it does not clearly judge whether the source post should mention the
  Muslim-convert identity.
- A comment that merely reports facts about the shooter, expresses sadness, or
  discusses the attack without taking a position on the relevance of the
  Muslim-convert identity is NEUTRAL (3).
- If the comment's position on the source framing is unclear, classify as 3.""",
            "support_rule": (
                "the comment clearly supports the source post's framing that "
                "Muslim-convert identity is relevant to report."
            ),
        }

    return {
        "discussion": "the source post's main claim or framing",
        "important": """- The source post presents the main claim or framing being discussed.
- Score 1 (Strongly Deny): The comment STRONGLY rejects, refutes, or argues
  against the source post's main claim or framing.
- Score 2 (Deny): The comment rejects, questions, or disagrees with the source
  post's main claim or framing, but less forcefully than score 1.
- Score 3 (Neutral/Question): The comment is neutral, unclear, asks a question,
  only reports facts, jokes/socializes, or does not take a clear stance toward
  the source post.
- Score 4 (Support): The comment supports, accepts, or agrees with the source
  post's main claim or framing.
- Score 5 (Strongly Support): The comment STRONGLY supports, explicitly
  confirms, or forcefully reinforces the source post's main claim or framing.""",
        "critical": """- Judge the TARGET COMMENT's stance toward the source post's main claim or
  framing.
- Merely mentioning the same topic as the source post is not enough for support
  or deny.
- A comment that only reports facts, jokes, socializes, asks unrelated
  questions, or lacks a clear stance is NEUTRAL (3).
- If the source-post relationship is unclear, classify as 3.""",
        "support_rule": (
            "the comment clearly supports the source post's main claim or "
            "framing."
        ),
    }


def build_stance_classification_prompt(
    source_text: str,
    comment_text: str,
    reply_context: str = "",
    mentioned_users: Optional[Sequence[str]] = None,
) -> str:
    """Build the canonical no-neighbor stance-classification prompt."""
    clean_source = str(source_text or "").strip()
    clean_comment = str(comment_text or "").strip()
    clean_context = str(reply_context or "").strip()
    clean_mentions = [
        str(user).strip()
        for user in (mentioned_users or [])
        if str(user).strip()
    ]

    if clean_context:
        context_block = (
            "\nMention context for reference only:\n"
            "The target comment mentions these users:\n"
            f"{', '.join('@' + user for user in clean_mentions)}\n\n"
            "The target comment is replying to or referring to the following "
            "earlier comments from those mentioned users. Only comments posted "
            "before the target comment are included.\n\n"
            "Use this context only to resolve what the TARGET COMMENT means: "
            "agreement, disagreement, pronouns, sarcasm, and references.\n"
            "Do not classify the stance of the mentioned users.\n"
            "Do not classify the context comments.\n"
            "Do not classify the conversation as a whole.\n\n"
            "Earlier mentioned-user comments:\n"
            f"{clean_context}\n"
        )
    else:
        context_block = ""

    prompt_spec = _stance_prompt_spec(clean_source)

    return f"""You are a strict stance classifier for social media discussions.

Determine the stance expressed by the TARGET COMMENT on the discussion about
{prompt_spec["discussion"]}.

Classify only the TARGET COMMENT. The source post and any context comments are
background only.

Stance scores:
{STANCE_SCORE_GUIDE}

IMPORTANT - what each score means for THIS thread:
{prompt_spec["important"]}

CRITICAL disambiguation rules:
{prompt_spec["critical"]}

Other rules:
- Judge the TARGET COMMENT specifically in relation to the source post.
- Consider sarcasm, negation, disagreement, and conversational context.
- Do not classify the source post itself.
- Do not let the stance of mentioned users or context comments override the
  stance expressed by the TARGET COMMENT.
- Classify as 4 or 5 only when {prompt_spec["support_rule"]}
- Return only one integer from 1 to 5.
- Do not provide an explanation.
- If the relationship is unclear, return 3.

Source post:
{clean_source}

TARGET COMMENT:
{clean_comment}
{context_block}

Return only one integer from 1 to 5.
"""


# extract_mentioned_screen_names 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def extract_mentioned_screen_names(
    text: str,
) -> List[str]:
    """Extract unique @mentioned screen names in text, preserving order."""
    names: List[str] = []
    seen = set()

    for match in re.finditer(
        r"@([A-Za-z0-9_]{1,20})",
        str(text or ""),
    ):
        name = match.group(1)
        key = name.lower()

        if key in seen:
            continue

        seen.add(key)
        names.append(name)

    return names


# build_mentioned_prior_comments_context 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_mentioned_prior_comments_context(
    *,
    comment_text: str,
    current_created_at: Optional[datetime],
    tweets: Dict[str, Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """Return earlier comments by users mentioned in the current comment."""
    mentioned_users = extract_mentioned_screen_names(
        comment_text
    )

    if not mentioned_users or current_created_at is None:
        return "", mentioned_users

    mentioned_keys = {
        user.lower()
        for user in mentioned_users
    }

    prior_rows: List[Tuple[datetime, str, str]] = []

    for tweet_id, tweet in tweets.items():
        created_at = get_created_at(
            tweet
        )

        if created_at is None or created_at >= current_created_at:
            continue

        screen_name = get_screen_name(
            tweet,
            str(tweet_id),
        )

        if screen_name.lower() not in mentioned_keys:
            continue

        prior_rows.append((
            created_at,
            screen_name,
            get_tweet_text(tweet),
        ))

    prior_rows.sort(
        key=lambda row: row[0]
    )

    if not prior_rows:
        return "", mentioned_users

    lines = []

    for created_at, screen_name, text in prior_rows:
        clean_text = (
            str(text or "")
            .replace('"', "'")
            .replace("\n", " ")
            .strip()
        )
        timestamp = created_at.isoformat()

        lines.append(
            f'- @{screen_name} at {timestamp}: "{clean_text}"'
        )

    return "\n".join(lines), mentioned_users


_INITIAL_CACHE: Dict[Tuple[str, str, str, bool], int] = {}


# llm_initial_opinion 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def llm_initial_opinion(
    text: str,
    source_text: str = "",
    reply_context: str = "",
    mentioned_users: Optional[Sequence[str]] = None,
    is_source: bool = False,
    use_llm: bool = True,
) -> int:
    """
    根据 source tweet 和当前评论，初始化评论 Agent 的立场。

    分类依据：
    - source_text：被讨论的 source tweet
    - text：当前 Agent 发表的评论

    source tweet 节点本身固定设为 4。
    """
    clean_text = str(text or "").strip()
    clean_source = str(source_text or "").strip()
    clean_reply_context = str(reply_context or "").strip()

    comment_preview = (
        clean_text
        .replace("\n", " ")
        [:120]
    )

    source_preview = (
        clean_source
        .replace("\n", " ")
        [:120]
    )

    # source tweet 节点本身固定为支持立场。
    if is_source:
        print(
            "[Init Opinion] LLM not called | "
            "reason=source_tweet | "
            f"score=4 | source={source_preview!r}"
        )
        return 4

    # 缓存必须同时考虑 source tweet 和评论原文。
    key = (
        clean_source,
        clean_text,
        clean_reply_context,
        is_source,
    )

    if key in _INITIAL_CACHE:
        cached_score = _INITIAL_CACHE[key]

        print(
            "[Init Opinion] LLM not called | "
            "reason=cache_hit | "
            f"score={cached_score} | "
            f"comment={comment_preview!r}"
        )

        return cached_score

    if not use_llm:
        score = heuristic_initial_opinion(
            clean_text,
            is_source=False,
        )

        _INITIAL_CACHE[key] = score

        print(
            "[Init Opinion] LLM not called | "
            "reason=use_llm_false | "
            f"heuristic_score={score} | "
            f"comment={comment_preview!r}"
        )

        return score

    if not API_KEY:
        score = heuristic_initial_opinion(
            clean_text,
            is_source=False,
        )

        _INITIAL_CACHE[key] = score

        print(
            "[Init Opinion] LLM not called | "
            "reason=missing_api_key | "
            f"heuristic_score={score} | "
            f"comment={comment_preview!r}"
        )

        return score

    prompt = build_stance_classification_prompt(
        clean_source,
        clean_text,
        reply_context=clean_reply_context,
        mentioned_users=mentioned_users,
    )

    print(
        "[Init Opinion] Calling LLM | "
        f"model={INIT_MODEL} | "
        f"source={source_preview!r} | "
        f"comment={comment_preview!r} | "
        f"mentioned_context={bool(clean_reply_context)}"
    )

    content, usage = llm_call(
        SYSTEM_MSG,
        prompt,
        model=INIT_MODEL,
        temperature=0.0,
        max_tokens=16,
    )

    print(
        "[Init Opinion] LLM returned | "
        f"response={content!r} | "
        f"tokens={usage.get('total_tokens', 0)}"
    )

    score = parse_opinion(content)

    if score is None:
        score = heuristic_initial_opinion(
            clean_text,
            is_source=False,
        )

        print(
            "[Init Opinion] LLM output parse failed | "
            f"fallback=heuristic | "
            f"score={score}"
        )
    else:
        print(
            "[Init Opinion] LLM score accepted | "
            f"score={score}"
        )

    _INITIAL_CACHE[key] = score

    print(
        "[Init Opinion] Final score | "
        f"score={score} | "
        f"comment={comment_preview!r}"
    )

    return score


# select_pheme_thread 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def select_pheme_thread(event_dir: Path, min_reactions: int = 10, thread_id: Optional[str] = None) -> Path:
    event_dir = Path(event_dir)
    if not event_dir.exists():
        raise FileNotFoundError(f"Event directory does not exist: {event_dir}")

    candidates: List[Tuple[int, str, Path]] = []
    for label_dir in ["rumours", "non-rumours"]:
        base = event_dir / label_dir
        if not base.exists():
            continue
        for td in base.iterdir():
            if not td.is_dir():
                continue
            if thread_id and td.name != str(thread_id):
                continue
            reaction_dir = td / "reactions"
            n_reactions = len(list(reaction_dir.glob("*.json"))) if reaction_dir.exists() else 0
            if thread_id:
                print(f"[PHEME] Selected specified thread: {td}")
                print(f"[PHEME] Label folder: {label_dir}")
                print(f"[PHEME] Reactions: {n_reactions}")
                return td
            if n_reactions >= min_reactions:
                candidates.append((n_reactions, label_dir, td))

    if thread_id:
        raise ValueError(f"Thread id {thread_id} not found under {event_dir}")
    if not candidates:
        raise ValueError(f"No thread found with >= {min_reactions} reactions in {event_dir}")

    candidates.sort(reverse=True, key=lambda x: x[0])
    n_reactions, label, td = candidates[0]
    print(f"[PHEME] Selected thread: {td}")
    print(f"[PHEME] Label folder: {label}")
    print(f"[PHEME] Reactions: {n_reactions}")
    return td


# load_pheme_thread 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def load_pheme_thread(
    thread_dir: Path,
    max_agents: int = 30,
    use_llm_init: bool = True,
    initial_opinions_override: Optional[Dict[int, int]] = None,
):
    """Convert a PHEME thread into graph, names, texts, initial opinions, and topic."""
    thread_dir = Path(thread_dir)
    source_dir = thread_dir / "source-tweet"
    reaction_dir = thread_dir / "reactions"
    structure_path = thread_dir / "structure.json"

    source_files = list(source_dir.glob("*.json"))
    if not source_files:
        raise ValueError(f"No source tweet found in {source_dir}")

    source_tweet = load_json(source_files[0])
    source_id = get_tweet_id(source_tweet, source_files[0].stem)
    tweets: Dict[str, Dict[str, Any]] = {source_id: source_tweet}

    if reaction_dir.exists():
        for f in reaction_dir.glob("*.json"):
            tw = load_json(f)
            tid = get_tweet_id(tw, f.stem)
            tweets[tid] = tw

    if structure_path.exists():
        edges = flatten_structure_tree(load_json(structure_path))
        print(f"[PHEME] Loaded edges from structure.json: {len(edges)}")
    else:
        edges = build_edges_from_reply_metadata(tweets, source_id)
        print(f"[PHEME] structure.json not found. Built edges from reply metadata: {len(edges)}")

    reaction_ids = [tid for tid in tweets if tid != source_id]
    reaction_ids.sort(key=lambda tid: get_created_at(tweets[tid]) or datetime.max)

    keep_ids = [source_id] + reaction_ids[: max_agents - 1]
    keep_set = set(keep_ids)
    id_map = {tid: i for i, tid in enumerate(keep_ids)}

    graph: Dict[int, List[int]] = {id_map[tid]: [] for tid in keep_ids}
    for u, v in edges:
        u, v = str(u), str(v)
        if u in keep_set and v in keep_set:
            ui, vi = id_map[u], id_map[v]
            if vi not in graph[ui]:
                graph[ui].append(vi)
            if ui not in graph[vi]:
                graph[vi].append(ui)

    source_idx = id_map[source_id]
    for tid in keep_ids:
        i = id_map[tid]
        if i == source_idx:
            continue
        if len(graph[i]) == 0:
            graph[i].append(source_idx)
            if i not in graph[source_idx]:
                graph[source_idx].append(i)

    agent_names: Dict[int, str] = {}
    agent_texts: Dict[int, str] = {}
    initial_opinions: Dict[int, int] = {}

    # 在初始化所有评论立场之前，先获得 source tweet 原文。
    topic = (
        get_tweet_text(source_tweet)
        or f"PHEME source tweet {source_id}"
    )

    for idx, tid in enumerate(keep_ids, start=1):
        print(f"[Init] Agent {idx}/{len(keep_ids)}")

        i = id_map[tid]
        tw = tweets[tid]

        agent_names[i] = get_screen_name(
            tw,
            tid,
        )

        agent_texts[i] = get_tweet_text(tw)

        if (
            initial_opinions_override is not None
            and i in initial_opinions_override
        ):
            initial_opinions[i] = int(
                initial_opinions_override[i]
            )
            print(
                "[Init Opinion] LLM not called | "
                "reason=stance_cache | "
                f"score={initial_opinions[i]} | "
                f"agent={i}"
            )
        else:
            reply_context, mentioned_users = (
                build_mentioned_prior_comments_context(
                    comment_text=agent_texts[i],
                    current_created_at=get_created_at(tw),
                    tweets=tweets,
                )
                if tid != source_id
                else ("", [])
            )

            initial_opinions[i] = llm_initial_opinion(
                text=agent_texts[i],
                source_text=topic,
                reply_context=reply_context,
                mentioned_users=mentioned_users,
                is_source=(tid == source_id),
                use_llm=use_llm_init,
            )
    for i in graph:
        graph[i] = sorted(graph[i])

    return graph, agent_names, agent_texts, initial_opinions, topic

# get_undirected_edges 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_undirected_edges(graph: Dict[int, List[int]]) -> List[Tuple[int, int]]:
    edges = set()
    for u, vs in graph.items():
        for v in vs:
            if u != v:
                edges.add(tuple(sorted((u, v))))
    return sorted(edges)


# build_role_groups 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_role_groups(graph: Dict[int, List[int]], frac: float = 0.2):
    degrees = {i: len(neigh) for i, neigh in graph.items()}
    sorted_nodes = sorted(degrees, key=lambda x: degrees[x], reverse=True)
    k = max(1, int(len(sorted_nodes) * frac))
    hub_nodes = sorted_nodes[:k]
    peripheral_nodes = sorted_nodes[-k:]
    return hub_nodes, peripheral_nodes, degrees


# summarize_graph 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def summarize_graph(graph: Dict[int, List[int]], degrees: Dict[int, int]) -> None:
    edges = get_undirected_edges(graph)
    avg_degree = sum(degrees.values()) / len(degrees) if degrees else 0
    print("\n[PHEME Graph Summary]")
    print(f"  Nodes/agents: {len(graph)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Max degree: {max(degrees.values()) if degrees else 0}")
    print("  Top-degree nodes:")
    for i in sorted(degrees, key=lambda x: degrees[x], reverse=True)[:10]:
        print(f"    {i:>3} @{AGENT_NAMES.get(i, f'agent_{i}'):<20} degree={degrees[i]}")


# ============================================================
# Prompt builders
# ============================================================

# safe_text 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def safe_text(text: str, n: int = 240) -> str:
    return text.replace('"', "'").replace("\n", " ")[:n]


# make_controlled_neighbor_lines 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def make_controlled_neighbor_lines(
    n_support: int = 0,
    n_deny: int = 0,
    n_neutral: int = 0,
    order: str = "alternating",
) -> List[str]:
    """Create synthetic but fixed neighbor posts for controlled experiments."""
    supports = [
        f'  - @support_user_{i}: Support (score 4/5). Post: "I agree with this claim. It seems correct and convincing."'
        for i in range(1, n_support + 1)
    ]
    denies = [
        f'  - @deny_user_{i}: Deny (score 2/5). Post: "I doubt this claim. It seems unconfirmed and questionable."'
        for i in range(1, n_deny + 1)
    ]
    neutrals = [
        f'  - @neutral_user_{i}: Question or Neutral (score 3/5). Post: "I am not sure yet. I need more evidence."'
        for i in range(1, n_neutral + 1)
    ]

    if order == "support_first":
        return supports + denies + neutrals
    if order == "deny_first":
        return denies + supports + neutrals

    # Alternating reduces a simple order confound.
    lines: List[str] = []
    max_len = max(len(supports), len(denies), len(neutrals))
    for i in range(max_len):
        if i < len(supports):
            lines.append(supports[i])
        if i < len(denies):
            lines.append(denies[i])
        if i < len(neutrals):
            lines.append(neutrals[i])
    return lines


# fmt_real_neighbor_opinions 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def fmt_real_neighbor_opinions(neighbors: List[int], opinions: Dict[int, int]) -> str:
    if not neighbors:
        return "(You have no visible replies or connected posts in this conversation.)"

    lines = []
    for n in neighbors[:MAX_NEIGHBORS_IN_PROMPT]:
        score = opinions[n]
        label = OPINION_LABELS[score]
        name = AGENT_NAMES.get(n, f"agent_{n}")
        text = safe_text(AGENT_TEXTS.get(n, ""), 180)
        lines.append(f'  - @{name}: {label} (score {score}/5). Post: "{text}"')
    return "\n".join(lines)


# build_prompt_controlled 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_prompt_controlled(
    agent_id: int,
    opinions: Dict[int, int],
    topic: str,
    neighbor_lines: List[str],
    role_label: Optional[str] = None,
    variant: str = "A",
) -> str:
    name = AGENT_NAMES.get(agent_id, f"agent_{agent_id}")
    my_score = opinions[agent_id]
    my_label = OPINION_LABELS[my_score]
    my_text = safe_text(AGENT_TEXTS.get(agent_id, ""), 240)
    role_str = f"You are known as a {role_label} in this conversation network.\n" if role_label else ""
    neighbor_str = "\n".join(neighbor_lines) if neighbor_lines else "(You have no visible replies or connected posts in this conversation.)"

    if variant == "B":
        return f"""You are @{name}, browsing a live social media rumor thread.
{role_str}The source post says: "{safe_text(topic, 500)}"

Your current reaction: {my_label} (score {my_score}/5)
Your own post: "{my_text}"

Here is what appears around you in the thread:
{neighbor_str}

After reading the thread context, what is your stance now?
Reply with ONLY your final opinion in this format: <number 1-5> - <label>"""

    if variant == "C":
        return f"""You are @{name}. Decide whether your stance should change.
{role_str}Source claim: "{safe_text(topic, 500)}"

Current stance: {my_score}/5, {my_label}
Your post: "{my_text}"

Nearby conversation signals:
{neighbor_str}

Return only the final stance as <number 1-5> - <label>."""

    return f"""You are @{name}, a participant in a rumor discussion.
{role_str}Rumor/source post being discussed: "{safe_text(topic, 500)}"

Your current stance toward the rumor: {my_label} (score {my_score}/5)
Your own post: "{my_text}"

Other connected posts in the conversation have the following stances:
{neighbor_str}

Based on these connected posts, what is your final stance toward the rumor?
Reply with ONLY your final opinion in this format: <number 1-5> - <label>"""


# build_prompt_real_graph 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_prompt_real_graph(
    agent_id: int,
    opinions: Dict[int, int],
    neighbors: List[int],
    topic: str,
    role_label: Optional[str] = None,
    variant: str = "A",
) -> str:
    neighbor_str = fmt_real_neighbor_opinions(neighbors, opinions)
    # Reuse the same wording as controlled prompts, but with real neighbor lines.
    return build_prompt_controlled(
        agent_id=agent_id,
        opinions=opinions,
        topic=topic,
        neighbor_lines=neighbor_str.split("\n") if neighbor_str else [],
        role_label=role_label,
        variant=variant,
    )


# ============================================================
# Experiment runners
# ============================================================
# get_opposite_candidate_ids 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_opposite_candidate_ids(
    agent_id: int,
    opinions: Dict[int, int],
) -> List[int]:
    """Return agents whose stance is opposite to the target agent."""
    old_score = opinions[agent_id]

    if old_score in {1, 2}:
        opposite_scores = {4, 5}
    elif old_score in {4, 5}:
        opposite_scores = {1, 2}
    else:
        opposite_scores = {1, 2, 4, 5}

    return [
        other_id
        for other_id, score in opinions.items()
        if (
            other_id != agent_id
            and other_id != 0
            and score in opposite_scores
        )
    ]


# get_support_candidate_ids 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def get_support_candidate_ids(
    agent_id: int,
    opinions: Dict[int, int],
) -> List[int]:
    """Return agents whose stance supports the target agent's current stance."""
    old_score = opinions[agent_id]

    if old_score in {1, 2}:
        support_scores = {1, 2}
    elif old_score in {4, 5}:
        support_scores = {4, 5}
    else:
        support_scores = {3}

    return [
        other_id
        for other_id, score in opinions.items()
        if (
            other_id != agent_id
            and other_id != 0
            and score in support_scores
        )
    ]


# build_threshold_neighbor_rankings 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_threshold_neighbor_rankings(
    graph: Dict[int, List[int]],
    opinions: Dict[int, int],
    *,
    target_agent_ids: Optional[Sequence[int]] = None,
    required_count: int = 5,
    required_support_count: int = 6,
    seed: int = 42,
) -> Tuple[
    Dict[int, Dict[str, List[Dict[str, Any]]]],
    List[int],
]:
    """Build fixed opposite/support neighbor rankings for each target agent."""
    if target_agent_ids is None:
        target_agent_ids = sorted(opinions)

    required_count = max(
        0,
        int(required_count),
    )
    required_support_count = max(
        0,
        int(required_support_count),
    )

    rankings: Dict[
        int,
        Dict[str, List[Dict[str, Any]]],
    ] = {}

    dropped_agents: List[int] = []

    # _rank_candidates 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
    def _rank_candidates(
        *,
        agent_id: int,
        eligible_ids: set,
        relation: str,
    ) -> List[Dict[str, Any]]:
        real_candidates = [
            neighbor_id
            for neighbor_id in graph.get(
                agent_id,
                [],
            )
            if neighbor_id in eligible_ids
        ]

        real_candidate_set = set(
            real_candidates
        )

        random_candidates = [
            candidate_id
            for candidate_id in eligible_ids
            if candidate_id not in real_candidate_set
        ]

        rng = random.Random(
            f"{seed}:{agent_id}:{relation}"
        )

        rng.shuffle(real_candidates)
        rng.shuffle(random_candidates)

        ordered = [
            {
                "agent_id": neighbor_id,
                "source": "real_graph",
                "relation": relation,
            }
            for neighbor_id in real_candidates
        ]

        ordered.extend(
            {
                "agent_id": neighbor_id,
                "source": "random_graph",
                "relation": relation,
            }
            for neighbor_id in random_candidates
        )

        return ordered

    for agent_id in target_agent_ids:
        if agent_id not in opinions:
            continue

        opposite_ids = set(
            get_opposite_candidate_ids(
                agent_id,
                opinions,
            )
        )

        support_ids = set(
            get_support_candidate_ids(
                agent_id,
                opinions,
            )
        )

        opposite_neighbors = _rank_candidates(
            agent_id=agent_id,
            eligible_ids=opposite_ids,
            relation="opposite",
        )

        support_neighbors = _rank_candidates(
            agent_id=agent_id,
            eligible_ids=support_ids,
            relation="support",
        )

        if (
            len(opposite_neighbors) < required_count
            or len(support_neighbors) < required_support_count
        ):
            dropped_agents.append(agent_id)
            continue

        rankings[agent_id] = {
            "opposite": opposite_neighbors,
            "support": support_neighbors,
        }

    return rankings, dropped_agents


# build_comment_threshold_prompt 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def build_comment_threshold_prompt(
    agent_id: int,
    neighbor_items: Sequence[Dict[str, Any]],
    *,
    agent_names: Dict[int, str],
    agent_texts: Dict[int, str],
    topic: str,
) -> str:
    """Build the threshold prompt that asks the agent to write a natural-language comment."""
    agent_name = agent_names.get(
        agent_id,
        f"agent_{agent_id}",
    )

    raw_own_comment = str(
        agent_texts.get(agent_id, "")
    ).strip()

    raw_topic = str(topic).strip()

    own_comment = (
        raw_own_comment
        .replace('"', "'")
        .replace("\n", " ")
        .strip()
    )

    clean_topic = (
        raw_topic
        .replace('"', "'")
        .replace("\n", " ")
        .strip()
    )

    if neighbor_items:
        neighbor_lines = []

        for item in neighbor_items:
            neighbor_id = int(
                item["agent_id"]
            )

            neighbor_name = agent_names.get(
                neighbor_id,
                f"agent_{neighbor_id}",
            )

            neighbor_comment = (
                str(
                    agent_texts.get(
                        neighbor_id,
                        "",
                    )
                )
                .replace('"', "'")
                .replace("\n", " ")
                .strip()
            )

            neighbor_lines.append(
                f'  - @{neighbor_name}: '
                f'"{neighbor_comment}"'
            )

        neighbor_text = "\n".join(
            neighbor_lines
        )
    else:
        neighbor_text = (
            "(No neighboring comments are visible.)"
        )

    return f"""You are @{agent_name}, a participant in a social media discussion.

Source post:
"{clean_topic}"

Your previously posted comment:
"{own_comment}"

The following original comments are currently visible to you:
{neighbor_text}

Write a natural-language follow-up comment that you would post now.
Base the comment only on the source post, your previous comment, and the visible original comments.
Your previously posted comment represents your current stance and speaking style.
Write as the same person continuing from that stance.

Important rules:
- Do not change your stance merely to sound polite, balanced, conciliatory, or agreeable.
- Only shift your stance if the visible comments give a concrete reason that would
  plausibly change your mind.
- If the visible comments conflict with each other, you may keep your prior stance,
  express uncertainty, or push back rather than seeking artificial compromise.
- If a visible comment uses the same display name as you, treat it as another
  original comment in the thread unless it is explicitly shown as your previously
  posted comment above.
- Do not output a stance score.
- Do not output a stance label such as Support, Deny, or Neutral.
- Do not explain the rating process.
- Write only the comment text.
- Keep the comment concise, like a social media reply."""

# run_comment_threshold_condition 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def run_comment_threshold_condition(
    condition_name: str,
    agent_ids: Sequence[int],
    opinions: Dict[int, int],
    topic: str,
    agent_names: Dict[int, str],
    agent_texts: Dict[int, str],
    neighbor_rankings: Dict[
        int,
        Dict[str, List[Dict[str, Any]]],
    ],
    neighbor_count: int,
    *,
    total_neighbor_count: int = 6,
    repetitions: int = 1,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    在真实评论回复图上运行阈值实验。

    每条评论是一个 Agent。

    邻居选择规则：
    - 优先使用真实图中的反对邻居；
    - 真实反对邻居不足时，使用随机补充邻居；
    - Prompt 只输入评论原文，不输入立场标签或分数。

    运行过程中会定期打印：
    - 已完成调用数量；
    - 当前进度百分比；
    - 有效结果数量；
    - 失败结果数量。
    """
    results: List[Dict[str, Any]] = []

    neighbor_count = max(
        0,
        int(neighbor_count),
    )

    total_neighbor_count = max(
        neighbor_count,
        int(total_neighbor_count),
    )


    # _call_agent 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
    def _call_agent(
        agent_id: int,
        rep: int,
    ) -> Dict[str, Any]:
        support_count = max(
            0,
            total_neighbor_count - neighbor_count,
        )
        ranking = neighbor_rankings[agent_id]
        opposite_items = ranking["opposite"][:neighbor_count]
        support_items = ranking["support"][:support_count]
        neighbor_items = opposite_items + support_items

        prompt = build_comment_threshold_prompt(
            agent_id=agent_id,
            neighbor_items=neighbor_items,
            agent_names=agent_names,
            agent_texts=agent_texts,
            topic=topic,
        )

        generated_comment, generation_usage = llm_call(
            (
                "You are simulating a specific social media user writing a concise "
                "natural-language reply. Continue from that user's prior stance and "
                "style. Do not become more agreeable merely for politeness. Do not "
                "output stance scores or stance labels."
            ),
            prompt,
            temperature=temperature,
            max_tokens=1024,
        )

        classification_prompt = build_stance_classification_prompt(
            topic,
            generated_comment,
        )

        classification_content, classification_usage = llm_call(
            SYSTEM_MSG,
            classification_prompt,
            temperature=0.0,
            max_tokens=16,
        )

        new_score = parse_opinion(
            classification_content
        )
        old_score = opinions[agent_id]
        generation_tokens = generation_usage.get(
            "total_tokens",
            0,
        )
        classification_tokens = classification_usage.get(
            "total_tokens",
            0,
        )

        return {
            "condition": condition_name,
            "agent": agent_id,
            "name": agent_names.get(
                agent_id,
                f"agent_{agent_id}",
            ),
            "old_score": old_score,
            "new_score": new_score,
            "change": (
                new_score - old_score
                if new_score is not None
                else None
            ),
            "generated_comment": generated_comment,
            "classification_response": classification_content,
            "raw_response": classification_content,
            "rep": rep,
            "tokens": generation_tokens + classification_tokens,
            "generation_tokens": generation_tokens,
            "classification_tokens": classification_tokens,
            "classification_prompt": classification_prompt,
            "neighbor_count": len(
                neighbor_items
            ),
            "opposite_neighbor_count": len(
                opposite_items
            ),
            "support_neighbor_count": len(
                support_items
            ),
            "neighbor_ids": [
                item["agent_id"]
                for item in neighbor_items
            ],
            "neighbor_sources": [
                item["source"]
                for item in neighbor_items
            ],
            "neighbor_relations": [
                item["relation"]
                for item in neighbor_items
            ],
            "real_neighbor_count": sum(
                item["source"] == "real_graph"
                for item in neighbor_items
            ),
            "random_neighbor_count": sum(
                item["source"] == "random_graph"
                for item in neighbor_items
            ),
            "prompt": prompt,
            "generation_prompt": prompt,
        }

    # 每个 Agent 在每次 repetition 中调用一次。
    tasks = [
        (agent_id, rep)
        for rep in range(repetitions)
        for agent_id in agent_ids
    ]

    total_tasks = len(tasks)
    completed_tasks = 0
    valid_tasks = 0
    failed_tasks = 0

    print(
        f"\n[{condition_name}] Starting "
        f"{total_tasks} tasks: "
        f"{len(agent_ids)} agents × "
        f"{repetitions} repetitions"
    )

    if total_tasks == 0:
        print(
            f"[{condition_name}] "
            "No tasks to run."
        )
        return results

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                _call_agent,
                agent_id,
                rep,
            ): (
                agent_id,
                rep,
            )
            for agent_id, rep in tasks
        }

        for future in as_completed(futures):
            agent_id, rep = futures[future]
            completed_tasks += 1

            try:
                result = future.result()
                results.append(result)

                if result.get("new_score") is not None:
                    valid_tasks += 1
                else:
                    failed_tasks += 1

            except Exception as error:
                failed_tasks += 1

                print(
                    f"\n[{condition_name}] "
                    f"Error: agent={agent_id}, "
                    f"rep={rep}, "
                    f"error={error}"
                )

            # 每完成 10 次调用打印一次进度。
            if (
                completed_tasks % 10 == 0
                or completed_tasks == total_tasks
            ):
                progress = (
                    completed_tasks
                    / total_tasks
                    * 100
                )

                print(
                    f"\r[{condition_name}] "
                    f"Task progress: "
                    f"{completed_tasks}/{total_tasks} "
                    f"({progress:.1f}%) | "
                    f"valid={valid_tasks} | "
                    f"failed={failed_tasks}",
                    end="",
                    flush=True,
                )

    # 结束当前行，避免后续输出和进度信息连在一起。
    print()

    results.sort(
        key=lambda result: (
            result["agent"],
            result["rep"],
        )
    )


    print(
        f"[{condition_name}] Finished | "
        f"tasks={len(results)} | "
        f"valid={valid_tasks} | "
        f"failed={failed_tasks}"
    )

    return results

# analyze_user_threshold_condition 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def analyze_user_threshold_condition(
    name: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    分析用户图阈值实验。

    对有明确极性的 Agent：
    - 原立场 1/2，分数上升表示向反方移动。
    - 原立场 4/5，分数下降表示向反方移动。

    对原立场为 3 的 Agent，只统计是否发生改变。
    """
    analysis = analyze_condition(
        name,
        results,
    )

    valid = [
        result
        for result in results
        if result.get("new_score") is not None
    ]

    polarized_results = [
        result
        for result in valid
        if result["old_score"] != 3
    ]

    # Count only true cross-side shifts. Same-side softening such as 1->2 or
    # 5->4 is not treated as movement to the opposite side.
    opposite_shift_count = sum(
        (
            result["old_score"] in {1, 2}
            and result["new_score"] in {4, 5}
        )
        or (
            result["old_score"] in {4, 5}
            and result["new_score"] in {1, 2}
        )
        for result in polarized_results
    )

    # Kept for backward-compatible reporting; this is now the same strict
    # cross-side criterion as opposite_shift_count.
    opposite_final_count = sum(
        (
            result["old_score"] in {1, 2}
            and result["new_score"] in {4, 5}
        )
        or (
            result["old_score"] in {4, 5}
            and result["new_score"] in {1, 2}
        )
        for result in polarized_results
    )

    neutral_results = [
        result
        for result in valid
        if result["old_score"] == 3
    ]

    analysis.update({
        "polarized_n": len(
            polarized_results
        ),
        "opposite_shift_count": (
            opposite_shift_count
        ),
        "opposite_shift_rate": (
            opposite_shift_count
            / len(polarized_results)
            if polarized_results
            else 0.0
        ),
        "opposite_final_count": (
            opposite_final_count
        ),
        "opposite_final_rate": (
            opposite_final_count
            / len(polarized_results)
            if polarized_results
            else 0.0
        ),
        "neutral_agent_n": len(
            neutral_results
        ),
        "neutral_agent_changed_rate": (
            sum(
                result["new_score"]
                != result["old_score"]
                for result in neutral_results
            )
            / len(neutral_results)
            if neutral_results
            else 0.0
        ),
        "avg_real_neighbor_count": (
            sum(
                result.get(
                    "real_neighbor_count",
                    0,
                )
                for result in valid
            )
            / len(valid)
            if valid
            else 0.0
        ),
        "avg_random_neighbor_count": (
            sum(
                result.get(
                    "random_neighbor_count",
                    0,
                )
                for result in valid
            )
            / len(valid)
            if valid
            else 0.0
        ),
        "avg_opposite_neighbor_count": (
            sum(
                result.get(
                    "opposite_neighbor_count",
                    0,
                )
                for result in valid
            )
            / len(valid)
            if valid
            else 0.0
        ),
        "avg_support_neighbor_count": (
            sum(
                result.get(
                    "support_neighbor_count",
                    0,
                )
                for result in valid
            )
            / len(valid)
            if valid
            else 0.0
        ),
    })

    return analysis

# run_controlled_condition 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def run_controlled_condition(
    condition_name: str,
    agent_ids: List[int],
    opinions: Dict[int, int],
    topic: str,
    neighbor_lines: List[str],
    prompt_variant: str = "A",
    role_labels: Optional[Dict[int, str]] = None,
    repetitions: int = 1,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # _call_agent 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
    def _call_agent(agent_id: int, rep: int):
        role_label = role_labels.get(agent_id) if role_labels else None
        prompt = build_prompt_controlled(
            agent_id=agent_id,
            opinions=opinions,
            topic=topic,
            neighbor_lines=neighbor_lines,
            role_label=role_label,
            variant=prompt_variant,
        )
        content, usage = llm_call(SYSTEM_MSG, prompt, temperature=temperature, max_tokens=60)
        new_score = parse_opinion(content)
        old_score = opinions[agent_id]
        return {
            "condition": condition_name,
            "agent": agent_id,
            "name": AGENT_NAMES.get(agent_id, f"agent_{agent_id}"),
            "old_score": old_score,
            "new_score": new_score,
            "change": (new_score - old_score) if new_score is not None else None,
            "raw_response": content,
            "rep": rep,
            "tokens": usage.get("total_tokens", 0),
            "neighbor_lines": neighbor_lines,
            "prompt_variant": prompt_variant,
        }

    tasks = [(aid, rep) for rep in range(repetitions) for aid in agent_ids]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_call_agent, aid, rep): (aid, rep) for aid, rep in tasks}
        for fut in as_completed(futures):
            aid, rep = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  Error in {condition_name}, agent={aid}, rep={rep}: {e}")

    results.sort(key=lambda x: (x["agent"], x["rep"]))
    return results


# run_comment_anchor_condition 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def run_comment_anchor_condition(
    condition_name: str,
    agent_ids: Sequence[int],
    opinions: Dict[int, int],
    topic: str,
    agent_names: Dict[int, str],
    agent_texts: Dict[int, str],
    neighbor_items: Sequence[Dict[str, Any]],
    *,
    repetitions: int = 1,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run anchoring with shared original comments, then classify generated comments."""
    results: List[Dict[str, Any]] = []
    neighbor_items = list(
        neighbor_items
    )

    support_neighbor_count = sum(
        item.get("relation") == "support"
        for item in neighbor_items
    )
    opposite_neighbor_count = sum(
        item.get("relation") == "opposite"
        for item in neighbor_items
    )

    # _call_agent 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
    def _call_agent(
        agent_id: int,
        rep: int,
    ) -> Dict[str, Any]:
        prompt = build_comment_threshold_prompt(
            agent_id=agent_id,
            neighbor_items=neighbor_items,
            agent_names=agent_names,
            agent_texts=agent_texts,
            topic=topic,
        )

        generated_comment, generation_usage = llm_call(
            (
                "You are simulating a specific social media user writing a concise "
                "natural-language reply. Continue from that user's prior stance and "
                "style. Do not become more agreeable merely for politeness. Do not "
                "output stance scores or stance labels."
            ),
            prompt,
            temperature=temperature,
            max_tokens=1024,
        )

        classification_prompt = build_stance_classification_prompt(
            topic,
            generated_comment,
        )

        classification_content, classification_usage = llm_call(
            SYSTEM_MSG,
            classification_prompt,
            temperature=0.0,
            max_tokens=16,
        )

        new_score = parse_opinion(
            classification_content
        )
        old_score = opinions[agent_id]
        generation_tokens = generation_usage.get(
            "total_tokens",
            0,
        )
        classification_tokens = classification_usage.get(
            "total_tokens",
            0,
        )

        return {
            "condition": condition_name,
            "agent": agent_id,
            "name": agent_names.get(
                agent_id,
                f"agent_{agent_id}",
            ),
            "old_score": old_score,
            "new_score": new_score,
            "change": (
                new_score - old_score
                if new_score is not None
                else None
            ),
            "generated_comment": generated_comment,
            "classification_response": classification_content,
            "raw_response": classification_content,
            "rep": rep,
            "tokens": generation_tokens + classification_tokens,
            "generation_tokens": generation_tokens,
            "classification_tokens": classification_tokens,
            "classification_prompt": classification_prompt,
            "neighbor_count": len(
                neighbor_items
            ),
            "opposite_neighbor_count": opposite_neighbor_count,
            "support_neighbor_count": support_neighbor_count,
            "neighbor_ids": [
                item["agent_id"]
                for item in neighbor_items
            ],
            "neighbor_sources": [
                item["source"]
                for item in neighbor_items
            ],
            "neighbor_relations": [
                item["relation"]
                for item in neighbor_items
            ],
            "real_neighbor_count": 0,
            "random_neighbor_count": 0,
            "prompt": prompt,
            "generation_prompt": prompt,
        }

    tasks = [
        (agent_id, rep)
        for rep in range(repetitions)
        for agent_id in agent_ids
    ]

    total_tasks = len(tasks)
    completed_tasks = 0
    valid_tasks = 0
    failed_tasks = 0

    print(
        f"\n[{condition_name}] Starting "
        f"{total_tasks} tasks: "
        f"{len(agent_ids)} agents x "
        f"{repetitions} repetitions"
    )

    if total_tasks == 0:
        print(
            f"[{condition_name}] "
            "No tasks to run."
        )
        return results

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:
        futures = {
            executor.submit(
                _call_agent,
                agent_id,
                rep,
            ): (
                agent_id,
                rep,
            )
            for agent_id, rep in tasks
        }

        for future in as_completed(futures):
            agent_id, rep = futures[future]
            completed_tasks += 1

            try:
                result = future.result()
                results.append(result)

                if result.get("new_score") is not None:
                    valid_tasks += 1
                else:
                    failed_tasks += 1

            except Exception as error:
                failed_tasks += 1

                print(
                    f"\n[{condition_name}] "
                    f"Error: agent={agent_id}, "
                    f"rep={rep}, "
                    f"error={error}"
                )

            if (
                completed_tasks % 10 == 0
                or completed_tasks == total_tasks
            ):
                progress = (
                    completed_tasks
                    / total_tasks
                    * 100
                )

                print(
                    f"\r[{condition_name}] "
                    f"Task progress: "
                    f"{completed_tasks}/{total_tasks} "
                    f"({progress:.1f}%) | "
                    f"valid={valid_tasks} | "
                    f"failed={failed_tasks}",
                    end="",
                    flush=True,
                )

    print()

    results.sort(
        key=lambda result: (
            result["agent"],
            result["rep"],
        )
    )

    print(
        f"[{condition_name}] Finished | "
        f"tasks={len(results)} | "
        f"valid={valid_tasks} | "
        f"failed={failed_tasks}"
    )

    return results


# run_real_graph_condition 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def run_real_graph_condition(
    condition_name: str,
    agent_ids: List[int],
    opinions: Dict[int, int],
    graph: Dict[int, List[int]],
    topic: str,
    prompt_variant: str = "A",
    role_labels: Optional[Dict[int, str]] = None,
    repetitions: int = 1,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # _call_agent 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
    def _call_agent(agent_id: int, rep: int):
        neighbors = graph.get(agent_id, [])
        role_label = role_labels.get(agent_id) if role_labels else None
        prompt = build_prompt_real_graph(
            agent_id=agent_id,
            opinions=opinions,
            neighbors=neighbors,
            topic=topic,
            role_label=role_label,
            variant=prompt_variant,
        )
        content, usage = llm_call(SYSTEM_MSG, prompt, temperature=temperature, max_tokens=60)
        new_score = parse_opinion(content)
        old_score = opinions[agent_id]
        return {
            "condition": condition_name,
            "agent": agent_id,
            "name": AGENT_NAMES.get(agent_id, f"agent_{agent_id}"),
            "old_score": old_score,
            "new_score": new_score,
            "change": (new_score - old_score) if new_score is not None else None,
            "raw_response": content,
            "rep": rep,
            "tokens": usage.get("total_tokens", 0),
            "degree": len(neighbors),
            "role_label": role_label,
            "prompt_variant": prompt_variant,
        }

    tasks = [(aid, rep) for rep in range(repetitions) for aid in agent_ids]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_call_agent, aid, rep): (aid, rep) for aid, rep in tasks}
        for fut in as_completed(futures):
            aid, rep = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  Error in {condition_name}, agent={aid}, rep={rep}: {e}")

    results.sort(key=lambda x: (x["agent"], x["rep"]))
    return results


# ============================================================
# Analysis
# ============================================================

# analyze_condition 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def analyze_condition(name: str, results: List[Dict[str, Any]], direction: Optional[str] = None) -> Dict[str, Any]:
    valid = [r for r in results if r.get("new_score") is not None]
    scores = [r["new_score"] for r in valid]
    changes = [r["change"] for r in valid]

    n = len(valid)
    n_changed = sum(1 for c in changes if c != 0)
    avg_abs_change = sum(abs(c) for c in changes) / n if n else 0
    avg_signed_change = sum(changes) / n if n else 0
    avg_final_score = sum(scores) / n if n else 0
    keep_rate = sum(1 for c in changes if c == 0) / n if n else 0
    neutral_rate = sum(1 for r in valid if r["new_score"] == 3) / n if n else 0
    support_rate = sum(1 for r in valid if r["new_score"] >= 4) / n if n else 0
    deny_rate = sum(1 for r in valid if r["new_score"] <= 2) / n if n else 0

    if direction == "up":
        directional_shift = sum(1 for r in valid if r["new_score"] > r["old_score"]) / n if n else 0
        target_rate = support_rate
    elif direction == "down":
        directional_shift = sum(1 for r in valid if r["new_score"] < r["old_score"]) / n if n else 0
        target_rate = deny_rate
    else:
        directional_shift = None
        target_rate = None

    if scores:
        mean_s = sum(scores) / len(scores)
        variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
    else:
        variance = 0

    call_change_by_initial_stance: Dict[int, Dict[str, Any]] = {}
    agent_change_by_initial_stance: Dict[int, Dict[str, Any]] = {}

    transition_counts = dict(
        sorted(
            Counter(
                (
                    r["old_score"],
                    r["new_score"],
                )
                for r in valid
            ).items()
        )
    )
    changed_results = [
        r
        for r in valid
        if r.get("change") != 0
    ]
    adjacent_change_count = sum(
        1
        for r in changed_results
        if abs(r["new_score"] - r["old_score"]) == 1
    )
    large_jump_count = sum(
        1
        for r in changed_results
        if abs(r["new_score"] - r["old_score"]) >= 2
    )
    polarity_flip_count = sum(
        (
            r["old_score"] in {1, 2}
            and r["new_score"] in {4, 5}
        )
        or (
            r["old_score"] in {4, 5}
            and r["new_score"] in {1, 2}
        )
        for r in changed_results
    )
    transition_examples: Dict[str, List[Dict[str, Any]]] = {}

    for r in changed_results:
        key = f"{r['old_score']}->{r['new_score']}"
        if len(transition_examples.get(key, [])) >= 3:
            continue

        transition_examples.setdefault(
            key,
            [],
        ).append({
            "agent": r.get("agent"),
            "rep": r.get("rep"),
            "old_score": r["old_score"],
            "new_score": r["new_score"],
            "change": r.get("change"),
            "generated_comment": r.get("generated_comment"),
            "raw_response": r.get("raw_response"),
        })

    transition_diagnostics = {
        "transition_counts": transition_counts,
        "changed_calls": len(changed_results),
        "adjacent_change_count": adjacent_change_count,
        "adjacent_change_rate": (
            adjacent_change_count / len(changed_results)
            if changed_results
            else 0.0
        ),
        "large_jump_count": large_jump_count,
        "large_jump_rate": (
            large_jump_count / len(changed_results)
            if changed_results
            else 0.0
        ),
        "polarity_flip_count": polarity_flip_count,
        "polarity_flip_rate": (
            polarity_flip_count / len(changed_results)
            if changed_results
            else 0.0
        ),
        "examples": transition_examples,
    }

    for stance in range(1, 6):
        stance_results = [
            r
            for r in valid
            if r.get("old_score") == stance
        ]
        changed_calls = sum(
            1
            for r in stance_results
            if r.get("change") != 0
        )

        call_change_by_initial_stance[stance] = {
            "valid_calls": len(stance_results),
            "changed_calls": changed_calls,
            "changed_call_rate": (
                changed_calls / len(stance_results)
                if stance_results
                else 0.0
            ),
        }

        agent_ids = sorted(
            {
                r["agent"]
                for r in stance_results
                if "agent" in r
            }
        )
        changed_agent_ids = sorted(
            {
                r["agent"]
                for r in stance_results
                if (
                    "agent" in r
                    and r.get("change") != 0
                )
            }
        )

        agent_change_by_initial_stance[stance] = {
            "agents": len(agent_ids),
            "changed_agents": len(changed_agent_ids),
            "changed_agent_rate": (
                len(changed_agent_ids) / len(agent_ids)
                if agent_ids
                else 0.0
            ),
            "changed_agent_ids": changed_agent_ids,
        }

    return {
        "name": name,
        "n": n,
        "changed": n_changed,
        "changed_rate": n_changed / n if n else 0,
        "avg_abs_change": avg_abs_change,
        "avg_signed_change": avg_signed_change,
        "avg_final_score": avg_final_score,
        "variance": variance,
        "keep_rate": keep_rate,
        "neutral_rate": neutral_rate,
        "support_rate": support_rate,
        "deny_rate": deny_rate,
        "directional_shift_rate": directional_shift,
        "target_rate": target_rate,
        "distribution": dict(sorted(Counter(scores).items())),
        "transition_diagnostics": transition_diagnostics,
        "call_change_by_initial_stance": call_change_by_initial_stance,
        "agent_change_by_initial_stance": agent_change_by_initial_stance,
        "tokens": sum(r.get("tokens", 0) for r in valid),
    }


# compare_pairwise 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def compare_pairwise(name: str, left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> Dict[str, Any]:
    lmap = {(r["agent"], r["rep"]): r for r in left if r.get("new_score") is not None}
    rmap = {(r["agent"], r["rep"]): r for r in right if r.get("new_score") is not None}
    keys = sorted(set(lmap) & set(rmap))

    if not keys:
        return {"name": name, "n_pairs": 0}

    diff_count = sum(1 for k in keys if lmap[k]["new_score"] != rmap[k]["new_score"])
    avg_abs_delta = sum(abs(rmap[k]["new_score"] - lmap[k]["new_score"]) for k in keys) / len(keys)
    right_more_change = sum(abs(rmap[k]["change"]) > abs(lmap[k]["change"]) for k in keys)
    left_more_change = sum(abs(lmap[k]["change"]) > abs(rmap[k]["change"]) for k in keys)

    return {
        "name": name,
        "n_pairs": len(keys),
        "diff_count": diff_count,
        "diff_rate": diff_count / len(keys),
        "avg_abs_final_score_delta": avg_abs_delta,
        "right_more_change_count": right_more_change,
        "left_more_change_count": left_more_change,
    }


# print_analysis 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def print_analysis(a: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"Condition: {a['name']}")
    print("=" * 72)
    print(f"  Valid calls:          {a['n']}")
    print(f"  Changed:              {a['changed']}/{a['n']} ({a['changed_rate']:.3f})")
    print(f"  Avg |change|:         {a['avg_abs_change']:.3f}")
    print(f"  Avg signed change:    {a['avg_signed_change']:.3f}")
    print(f"  Avg final score:      {a['avg_final_score']:.3f}")
    print(f"  Keep current rate:    {a['keep_rate']:.3f}")
    print(f"  Neutral final rate:   {a['neutral_rate']:.3f}")
    print(f"  Support final rate:   {a['support_rate']:.3f}")
    print(f"  Deny final rate:      {a['deny_rate']:.3f}")
    if a.get("directional_shift_rate") is not None:
        print(f"  Directional shift:    {a['directional_shift_rate']:.3f}")
        print(f"  Target final rate:    {a['target_rate']:.3f}")
    print(f"  Variance:             {a['variance']:.3f}")
    print(f"  Distribution:         {a['distribution']}")
    if "transition_diagnostics" in a:
        d = a["transition_diagnostics"]
        print("  Transition diagnostics:")
        print(
            f"    Adjacent changes:   "
            f"{d['adjacent_change_count']}/"
            f"{d['changed_calls']} "
            f"({d['adjacent_change_rate']:.3f})"
        )
        print(
            f"    Large jumps >=2:    "
            f"{d['large_jump_count']}/"
            f"{d['changed_calls']} "
            f"({d['large_jump_rate']:.3f})"
        )
        print(
            f"    Polarity flips:     "
            f"{d['polarity_flip_count']}/"
            f"{d['changed_calls']} "
            f"({d['polarity_flip_rate']:.3f})"
        )
        print("    Transition counts:")
        for (old_score, new_score), count in d[
            "transition_counts"
        ].items():
            if old_score == new_score:
                continue
            print(
                f"      {old_score}->{new_score}: "
                f"{count}"
            )
    if "agent_change_by_initial_stance" in a:
        print("  Changed agents by initial stance:")
        for stance, row in a["agent_change_by_initial_stance"].items():
            print(
                f"    score {stance}: "
                f"{row['changed_agents']}/{row['agents']} "
                f"({row['changed_agent_rate']:.3f})"
            )
    if "call_change_by_initial_stance" in a:
        print("  Changed calls by initial stance:")
        for stance, row in a["call_change_by_initial_stance"].items():
            print(
                f"    score {stance}: "
                f"{row['changed_calls']}/{row['valid_calls']} "
                f"({row['changed_call_rate']:.3f})"
            )
    print(f"  Tokens:               {a['tokens']}")


# print_generated_comment_samples 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def print_generated_comment_samples(
    results: Sequence[Dict[str, Any]],
    *,
    limit: int = 5,
) -> None:
    """Print a small sample of generated natural-language comments."""
    rows = [
        result
        for result in results
        if result.get("generated_comment")
    ]

    if not rows:
        print("  Generated comments:   none")
        return

    print(
        f"  Generated comment samples "
        f"(showing {min(limit, len(rows))}/{len(rows)}):"
    )

    for result in rows[:limit]:
        comment = (
            str(result.get("generated_comment", ""))
            .replace("\n", " ")
            .strip()
        )

        print(
            "    "
            f"agent={result.get('agent')} "
            f"rep={result.get('rep')} "
            f"{result.get('old_score')}->{result.get('new_score')} | "
            f"{comment}"
        )


# print_pairwise 函数，供 PHEME 实验加载数据、构造 prompt、运行 LLM 或统计结果时调用。
def print_pairwise(c: Dict[str, Any]) -> None:
    print("\n" + "-" * 72)
    print(f"Pairwise comparison: {c['name']}")
    print("-" * 72)
    if c.get("n_pairs", 0) == 0:
        print("  No matched pairs.")
        return
    print(f"  Matched pairs:        {c['n_pairs']}")
    print(f"  Different outputs:    {c['diff_count']}/{c['n_pairs']} ({c['diff_rate']:.3f})")
    print(f"  Avg final-score delta:{c['avg_abs_final_score_delta']:.3f}")
    print(f"  Right more changed:   {c['right_more_change_count']}")
    print(f"  Left more changed:    {c['left_more_change_count']}")


# ============================================================
# Agent selection
# ============================================================

