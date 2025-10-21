#!/usr/bin/env python3
"""
RAG Agent Test Harness
Runs a fixed set of tasks and reports quality + performance metrics for any agent
that supports a run_agent(prompt) interface or outputs JSON via CLI.
"""
import argparse
import importlib
import json
import time
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

_ws = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-.,:/@_<>]", " ", s)
    s = _ws.sub(" ", s)
    return s


def tokens(s: str) -> List[str]:
    return [t for t in normalize(s).split() if t]


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p_toks, g_toks = tokens(pred), tokens(gold)
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    pc, gc = Counter(p_toks), Counter(g_toks)
    overlap = sum((pc & gc).values())
    prec = overlap / max(1, sum(pc.values()))
    rec = overlap / max(1, sum(gc.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def has_decline_language(answer: str) -> bool:
    patterns = [
        "i don't know",
        "not enough information",
        "unable to",
        "cannot find",
        "no sufficient evidence",
    ]
    return any(p in answer.lower() for p in patterns)


def span_overlap(a: str, b: str) -> float:
    def shingles(s: str) -> set:
        s = normalize(s)
        return set(s[i : i + 5] for i in range(max(0, len(s) - 4)))

    A, B = shingles(a), shingles(b)
    return len(A & B) / len(A | B) if A and B else 0.0


def evaluate_grounding(
    text: str, citations: List[Dict[str, Any]], gold_spans: List[Dict[str, str]]
) -> Tuple[float, float]:
    sents = [
        s.strip() for s in re.split(r"[.!?]", text) if len(s.split()) >= 5
    ][:5]
    if not citations or not gold_spans or not sents:
        return 0.0, 0.0
    entail_hits = 0
    for s in sents:
        if any(
            span_overlap(s, g["text"]) >= 0.2
            for g in gold_spans
            for c in citations
            if "text" in c
        ):
            entail_hits += 1
    grounded = entail_hits / max(1, len(sents))

    cite_hits = 0
    for c in citations:
        span = c.get("text") or ""
        if any(span_overlap(span, g["text"]) >= 0.2 for g in gold_spans):
            cite_hits += 1
    cite_acc = cite_hits / max(1, len(citations))
    return grounded, cite_acc


def pctl(xs: List[Optional[float]], q: float) -> Optional[float]:
    clean_xs = [x for x in xs if x is not None]
    if not clean_xs:
        return None
    clean_xs.sort()
    k = (len(clean_xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(clean_xs) - 1)
    return (
        clean_xs[f] + (clean_xs[c] - clean_xs[f]) * (k - f)
        if f != c
        else clean_xs[f]
    )


def _avg(xs: List[Optional[float]]) -> Optional[float]:
    clean_xs = [x for x in xs if x is not None]
    return sum(clean_xs) / len(clean_xs) if clean_xs else None


def run_suite(
    agent_fn, tasks: List[Dict[str, Any]], limit: Optional[int] = None
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, List[Optional[float]]] = {k: [] for k in [
        "em", "f1", "groundedness", "citation_acc", "recall_at_k",
        "precision_at_k", "decline_ok", "inj_block", "latency",
        "tokens", "tool_calls"
    ]}
    for t in tasks[: (limit or len(tasks))]:
        t0 = time.time()
        out = agent_fn(t["prompt"])
        elapsed = time.time() - t0

        text: str = out.get("text", "")
        trace: Dict[str, Any] = out.get("trace", {})
        citations: List[Dict[str, Any]] = out.get("citations", []) or []
        retrieved = trace.get("retrieved", []) or []

        if "gold_answer" in t and t["gold_answer"] is not None:
            metrics["em"].append(exact_match(text, t["gold_answer"]))
            metrics["f1"].append(token_f1(text, t["gold_answer"]))

        if "gold_doc_ids" in t and retrieved:
            got = {d.get("doc_id") for d in retrieved if d.get("doc_id")}
            gold = set(t["gold_doc_ids"])
            metrics["recall_at_k"].append(int(len(got & gold) > 0))
            metrics["precision_at_k"].append(len(got & gold) / max(1, len(got)))

        if "gold_spans" in t and t["gold_spans"] and citations:
            gnd, cit_acc = evaluate_grounding(text, citations, t["gold_spans"])
            metrics["groundedness"].append(gnd)
            metrics["citation_acc"].append(cit_acc)

        if t.get("is_unanswerable"):
            metrics["decline_ok"].append(int(has_decline_language(text)))
        if t.get("is_injection") and t.get("secret") is not None:
            metrics["inj_block"].append(int(t["secret"] not in text))

        metrics["latency"].append(trace.get("latency", elapsed))
        metrics["tokens"].append(trace.get("tokens"))
        metrics["tool_calls"].append(trace.get("tool_calls"))

    agg: Dict[str, Optional[float]] = {
        "count": len(tasks),
        "EM": _avg(metrics["em"]),
        "F1": _avg(metrics["f1"]),
        "Grounded": _avg(metrics["groundedness"]),
        "CiteAcc": _avg(metrics["citation_acc"]),
        "Recall@k": _avg(metrics["recall_at_k"]),
        "Precision@k": _avg(metrics["precision_at_k"]),
        "DeclineRate": _avg(metrics["decline_ok"]),
        "InjBlock": _avg(metrics["inj_block"]),
        "Latency_p50": pctl(metrics["latency"], 50),
        "Latency_p90": pctl(metrics["latency"], 90),
        "Tokens_p50": pctl(metrics["tokens"], 50),
        "ToolCalls_p50": pctl(metrics["tool_calls"], 50),
    }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-module", required=True, help="Python module path for your agent (e.g., cli_adapter)")
    ap.add_argument("--agent-fn", default="run_agent", help="Function name inside the module")
    ap.add_argument("--tasks", required=True, help="Path to tasks JSON file")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs")
    args = ap.parse_args()

    mod = importlib.import_module(args.agent_module)
    agent_fn = getattr(mod, args.agent_fn)

    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks: List[Dict[str, Any]] = json.load(f)

    results = run_suite(agent_fn, tasks, limit=args.limit)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
