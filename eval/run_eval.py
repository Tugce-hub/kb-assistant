"""Offline evaluation harness.

    python eval/run_eval.py retrieval                    # all retrieval modes, default models
    python eval/run_eval.py retrieval --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --tag multilingual
    python eval/run_eval.py generate                     # end-to-end answers + LLM judge (needs ANTHROPIC_API_KEY)
    python eval/run_eval.py report                       # render tables into README.md
    python eval/run_eval.py gate --min-hit5 0.85         # CI regression gate on retrieval

Retrieval metrics need no API key and run in CI on every PR.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from kbassist.config import Settings, get_settings  # noqa: E402
from kbassist.index.store import build_index, load_index, save_index  # noqa: E402
from kbassist.retrieval import MODES, Retriever  # noqa: E402
from kbassist.security.acl import ACL  # noqa: E402

RESULTS = ROOT / "eval" / "results"
EVAL_USER = "eval-runner"
KS = (1, 3, 5)


def load_questions() -> list[dict]:
    return yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))["questions"]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[idx]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def with_overrides(
    settings: Settings,
    embedding_model: str | None,
    reranker_model: str | None,
    rerank_pool: int | None = None,
    rerank_max_chars: int | None = None,
) -> tuple[Settings, Path]:
    raw = json.loads(json.dumps(settings.raw))
    index_dir = settings.path("paths.index_dir")
    if embedding_model and embedding_model != raw["embedding"]["model"]:
        raw["embedding"]["model"] = embedding_model
        index_dir = index_dir.with_name(f"index__{slug(embedding_model)}")
    if reranker_model:
        raw["reranker"]["model"] = reranker_model
    if rerank_pool:
        raw["retrieval"]["rerank_pool"] = rerank_pool
    if rerank_max_chars:
        raw["retrieval"]["rerank_max_chars"] = rerank_max_chars
    return replace(settings, raw=raw), index_dir


# --------------------------------------------------------------------------- retrieval


def score_ranking(paths: list[str], gold: list[str]) -> dict:
    gold_set = set(gold)
    first = next((i for i, p in enumerate(paths, start=1) if p in gold_set), None)
    out = {"rr": 1.0 / first if first else 0.0, "first_gold_rank": first}
    for k in KS:
        top = set(paths[:k])
        out[f"hit@{k}"] = float(bool(top & gold_set))
        out[f"recall@{k}"] = len(top & gold_set) / len(gold_set)
    return out


def aggregate(rows: list[dict]) -> dict:
    agg = {key: statistics.mean(r[key] for r in rows) for key in rows[0] if key.startswith(("hit@", "recall@"))}
    agg["mrr@10"] = statistics.mean(r["rr"] for r in rows)
    agg["n"] = len(rows)
    lat = [r["latency_ms"] for r in rows]
    agg["p50_ms"] = pct(lat, 50)
    agg["p95_ms"] = pct(lat, 95)
    return agg


def run_retrieval(args) -> None:
    base = get_settings()
    settings, index_dir = with_overrides(
        base, args.embedding_model, args.reranker_model, args.rerank_pool, args.rerank_max_chars
    )
    if (index_dir / "manifest.json").exists():
        index = load_index(index_dir)
    else:
        print(f"building index for {settings.get('embedding.model')} -> {index_dir}")
        index = build_index(settings, sync=False)
        save_index(index, index_dir)
    acl = ACL.from_file(settings.path("acl_file"))
    retriever = Retriever(index, settings, acl)
    principal = acl.principal(EVAL_USER)
    questions = [q for q in load_questions() if q["gold"]]
    modes = args.modes.split(",") if args.modes else list(MODES)

    report = {
        "tag": args.tag,
        "embedding_model": settings.get("embedding.model"),
        "reranker_model": settings.get("reranker.model"),
        "rerank_pool": settings.get("retrieval.rerank_pool"),
        "rerank_max_chars": settings.get("retrieval.rerank_max_chars"),
        "num_chunks": len(index.chunks),
        "commit": index.manifest["commits"],
        "modes": {},
    }
    for mode in modes:
        retriever.search("warm up the models", principal, mode=mode, k=10)
        rows = []
        for q in questions:
            t0 = time.perf_counter()
            hits = retriever.search(q["question"], principal, mode=mode, k=10)
            latency = (time.perf_counter() - t0) * 1000
            paths = [h.chunk.path for h in hits]
            rows.append({"id": q["id"], "lang": q["lang"], "category": q["category"], "latency_ms": latency,
                         "paths": paths, **score_ranking(paths, q["gold"])})
        by: dict[str, dict] = {}
        for field in ("category", "lang"):
            for value in sorted({r[field] for r in rows}):
                by[f"{field}={value}"] = aggregate([r for r in rows if r[field] == value])
        report["modes"][mode] = {"overall": aggregate(rows), "breakdown": by, "per_question": rows}
        o = report["modes"][mode]["overall"]
        print(f"{mode:15s} hit@1={o['hit@1']:.3f} hit@5={o['hit@5']:.3f} recall@5={o['recall@5']:.3f} "
              f"mrr={o['mrr@10']:.3f} p50={o['p50_ms']:.0f}ms p95={o['p95_ms']:.0f}ms")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"retrieval__{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")


# -------------------------------------------------------------------------- generation


def run_generate(args) -> None:
    from judge import judge
    from kbassist.llm import make_llm
    from kbassist.service import KnowledgeService

    svc = KnowledgeService()
    judge_llm = make_llm(svc.settings, "judge")
    judge_model = judge_llm.model
    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    svc.search("warm up", EVAL_USER, k=1)

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        answer, trace = svc.ask(q["question"], EVAL_USER, channel="eval")
        e2e = (time.perf_counter() - t0) * 1000
        verdict, judge_usage = judge(judge_llm, q["question"], q["reference"], answer.hits, answer.text)
        retrieved = [h.chunk.path for h in answer.hits]
        cited = [h.chunk.path for h in answer.cited_hits]
        row = {
            "id": q["id"], "lang": q["lang"], "category": q["category"],
            "answerable_expected": bool(q["gold"]),
            "answerable_pred": answer.answerable,
            "abstained": verdict.abstained or not answer.answerable,
            "correctness": verdict.correctness,
            "n_claims": len(verdict.claims),
            "n_unsupported": len(verdict.unsupported_claims),
            "unsupported_claims": verdict.unsupported_claims,
            "rationale": verdict.rationale,
            "gold_retrieved": bool(set(retrieved) & set(q["gold"])),
            "cites_gold": bool(set(cited) & set(q["gold"])),
            "latency_ms": e2e,
            "retrieval_ms": trace.spans_ms.get("retrieval_total", 0.0),
            "llm_ms": trace.spans_ms.get("llm", 0.0),
            "input_tokens": trace.usage.input_tokens,
            "output_tokens": trace.usage.output_tokens,
            "cost_usd": trace.usage.cost_usd(svc.pricing),
            "judge_cost_usd": judge_usage.cost_usd(svc.pricing),
            "model": answer.model,
            "answer": answer.text,
        }
        rows.append(row)
        print(f"{q['id']}: {row['correctness']:9s} abst={row['abstained']!s:5s} unsup={row['n_unsupported']}/"
              f"{row['n_claims']} {e2e/1000:.1f}s ${row['cost_usd']:.4f}")

    answerable = [r for r in rows if r["answerable_expected"]]
    unanswerable = [r for r in rows if not r["answerable_expected"]]
    answered = [r for r in rows if not r["abstained"]]
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    summary = {
        "provider": svc.settings.get("llm.provider"),
        "model": svc.generator.llm.model,
        "effort": svc.settings.get(f"llm.{svc.settings.get('llm.provider')}.effort"),
        "judge_model": judge_model,
        "n": len(rows),
        "answer_accuracy": statistics.mean(score[r["correctness"]] for r in answerable) if answerable else None,
        "strict_correct_rate": statistics.mean(r["correctness"] == "correct" for r in answerable) if answerable else None,
        "hallucination_rate": statistics.mean(r["n_unsupported"] > 0 for r in answered) if answered else None,
        "unsupported_claim_rate": (sum(r["n_unsupported"] for r in answered) / max(1, sum(r["n_claims"] for r in answered))),
        "abstention_recall": statistics.mean(r["abstained"] for r in unanswerable) if unanswerable else None,
        "false_abstention_rate": statistics.mean(r["abstained"] for r in answerable) if answerable else None,
        "citation_gold_rate": statistics.mean(r["cites_gold"] for r in answerable if not r["abstained"]) if answerable else None,
        "latency_p50_ms": pct([r["latency_ms"] for r in rows], 50),
        "latency_p95_ms": pct([r["latency_ms"] for r in rows], 95),
        "retrieval_p50_ms": pct([r["retrieval_ms"] for r in rows], 50),
        "llm_p50_ms": pct([r["llm_ms"] for r in rows], 50),
        "llm_p95_ms": pct([r["llm_ms"] for r in rows], 95),
        "mean_input_tokens": statistics.mean(r["input_tokens"] for r in rows),
        "mean_output_tokens": statistics.mean(r["output_tokens"] for r in rows),
        "cost_per_query_usd": statistics.mean(r["cost_usd"] for r in rows),
        "cost_per_1k_queries_usd": 1000 * statistics.mean(r["cost_usd"] for r in rows),
        "judge_cost_total_usd": sum(r["judge_cost_usd"] for r in rows),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "generation.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------------------------ report


def _fmt(v, kind="f") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "ms":
        return f"{v:.1f} ms" if v < 10 else f"{v:,.0f} ms"
    if kind == "usd":
        return f"${v:.4f}"
    return f"{v:.3f}"


MODE_LABEL = {
    "bm25": "BM25 only",
    "vector": "Vector only",
    "hybrid": "Hybrid (BM25 + vector, RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}


def render_retrieval(main: dict) -> str:
    lines = [
        f"Embedding `{main['embedding_model']}` · reranker `{main['reranker_model']}` · "
        f"{main['modes'][next(iter(main['modes']))]['overall']['n']} answerable questions · {main['num_chunks']} chunks",
        "",
        "| Retrieval mode | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@10 | p50 latency | p95 latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, data in main["modes"].items():
        o = data["overall"]
        lines.append(
            f"| {MODE_LABEL.get(mode, mode)} | {_fmt(o['hit@1'])} | {_fmt(o['hit@3'])} | {_fmt(o['hit@5'])} | "
            f"{_fmt(o['recall@5'])} | {_fmt(o['mrr@10'])} | {_fmt(o['p50_ms'], 'ms')} | {_fmt(o['p95_ms'], 'ms')} |"
        )
    return "\n".join(lines)


def render_breakdown(main: dict) -> str:
    modes = list(main["modes"])
    keys = list(main["modes"][modes[0]]["breakdown"])
    lines = ["| Slice | n | " + " | ".join(f"{MODE_LABEL.get(m, m)} Hit@5" for m in modes) + " |",
             "|---|---:|" + "---:|" * len(modes)]
    for key in keys:
        n = main["modes"][modes[0]]["breakdown"][key]["n"]
        cells = " | ".join(_fmt(main["modes"][m]["breakdown"][key]["hit@5"]) for m in modes)
        lines.append(f"| {key.replace('=', ': ')} | {n} | {cells} |")
    return "\n".join(lines)


def render_ablation(reports: list[dict]) -> str:
    lines = ["| Embedding model | Reranker | Vector Hit@5 | Hybrid Hit@5 | Hybrid+rerank Hit@5 | Hybrid+rerank MRR@10 "
             "| Turkish Hit@5 (hybrid / +rerank) | Hybrid+rerank p50 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        m = r["modes"]
        get = lambda mode, key="hit@5", sl=None: (  # noqa: E731
            (m[mode]["breakdown"].get(sl, {}) if sl else m[mode]["overall"]).get(key) if mode in m else None
        )
        lines.append(
            f"| {'**chosen** ' if r['tag'] == 'default' else ''}`{r['embedding_model']}` | `{r['reranker_model']}` (pool {r.get('rerank_pool', 20)}"
            f"{', ' + str(r['rerank_max_chars']) + ' chars' if r.get('rerank_max_chars') else ''}) | "
            f"{_fmt(get('vector'))} | {_fmt(get('hybrid'))} | "
            f"{_fmt(get('hybrid_rerank'))} | {_fmt(get('hybrid_rerank', 'mrr@10'))} | "
            f"{_fmt(get('hybrid', sl='lang=tr'))} / {_fmt(get('hybrid_rerank', sl='lang=tr'))} | "
            f"{_fmt(get('hybrid_rerank', 'p50_ms'), 'ms')} |"
        )
    return "\n".join(lines)


SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def audited_summary(gen: dict, audit: dict) -> dict:
    """Recompute the quality metrics from second-pass labels, plus judge agreement."""
    labels = audit["labels"]
    rows = [r for r in gen["rows"] if r["id"] in labels]
    answerable = [r for r in rows if r["answerable_expected"]]
    unanswerable = [r for r in rows if not r["answerable_expected"]]
    answered = [r for r in rows if not r["abstained"]]
    lab = lambda r, k, d=False: labels[r["id"]].get(k, d)  # noqa: E731
    return {
        "answer_accuracy": statistics.mean(SCORE[lab(r, "correctness")] for r in answerable),
        "strict_correct_rate": statistics.mean(lab(r, "correctness") == "correct" for r in answerable),
        "hallucination_rate": statistics.mean(bool(lab(r, "hallucination")) for r in answered),
        "abstention_recall": statistics.mean(lab(r, "correctness") == "correct" for r in unanswerable),
        "false_abstention_rate": statistics.mean(bool(lab(r, "false_abstain")) for r in answerable),
        "judge_correctness_agreement": statistics.mean(r["correctness"] == lab(r, "correctness") for r in rows),
        "judge_hallucination_agreement": statistics.mean((r["n_unsupported"] > 0) == bool(lab(r, "hallucination")) for r in answered),
        "n_reviewed": len(rows),
    }


def render_generation(gen: dict, audit: dict | None = None) -> str:
    s = gen["summary"]
    a = audited_summary(gen, audit) if audit else {}
    pct = lambda d, k: _fmt(d.get(k), "pct")  # noqa: E731
    quality = [
        ("Answer accuracy (correct = 1, partial = 0.5)", "answer_accuracy"),
        ("Strictly correct answers", "strict_correct_rate"),
        ("**Hallucination rate** (answers with a claim not supported by the sources)", "hallucination_rate"),
        ("Declines unanswerable questions (higher is better)", "abstention_recall"),
        ("False \"not in the sources\" on answerable questions (lower is better)", "false_abstention_rate"),
    ]
    lines = ["| Quality metric | Local 3B judge (automatic) | Second-pass review |", "|---|---:|---:|"]
    lines += [f"| {label} | {pct(s, key)} | {pct(a, key) if a else '–'} |" for label, key in quality]
    if a:
        lines.append(f"| Judge agrees with review: correctness / hallucination flag | "
                     f"{pct(a, 'judge_correctness_agreement')} / {pct(a, 'judge_hallucination_agreement')} | |")
    ops = [
        ("Answers citing a gold file", _fmt(s["citation_gold_rate"], "pct")),
        ("End-to-end latency p50 / p95", f"{_fmt(s['latency_p50_ms'], 'ms')} / {_fmt(s['latency_p95_ms'], 'ms')}"),
        ("  of which retrieval p50", _fmt(s["retrieval_p50_ms"], "ms")),
        ("  of which LLM p50 / p95", f"{_fmt(s['llm_p50_ms'], 'ms')} / {_fmt(s['llm_p95_ms'], 'ms')}"),
        ("Mean tokens in / out", f"{s['mean_input_tokens']:,.0f} / {s['mean_output_tokens']:,.0f}"),
        ("**Cost per query** / per 1k queries", f"{_fmt(s['cost_per_query_usd'], 'usd')} / ${s['cost_per_1k_queries_usd']:.2f}"),
    ]
    where = "local via Ollama, CPU only" if s.get("provider") == "ollama" else f"Claude API, effort `{s['effort']}`"
    head = f"Generator `{s['model']}` ({where}), judge `{s['judge_model']}`, n = {s['n']}"
    if a:
        head += f"; second-pass labels: `eval/results/audit.yaml` ({audit.get('reviewer', '')})"
    return (head + "\n\n" + "\n".join(lines) + "\n\n| Operational metric | Value |\n|---|---:|\n"
            + "\n".join(f"| {x} | {y} |" for x, y in ops))


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- BEGIN:{name} -->\n).*?(<!-- END:{name} -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README is missing the {name} markers")
    return pattern.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), text)


def run_report(args) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    main_path = RESULTS / "retrieval__default.json"
    if main_path.exists():
        main = json.loads(main_path.read_text(encoding="utf-8"))
        text = replace_block(text, "retrieval", render_retrieval(main))
        text = replace_block(text, "breakdown", render_breakdown(main))
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("retrieval__*.json"))]
    if reports:
        # Experiment order (how the choice was reached), chosen config last.
        order = ["baseline-en", "jina-reranker", "multilingual", "multilingual-pool20-600c", "default"]
        reports.sort(key=lambda r: order.index(r["tag"]) if r["tag"] in order else len(order) - 1)
        text = replace_block(text, "ablation", render_ablation(reports))
    gen_path = RESULTS / "generation.json"
    if gen_path.exists():
        audit_path = RESULTS / "audit.yaml"
        audit = yaml.safe_load(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        text = replace_block(text, "generation", render_generation(gen, audit))
    readme.write_text(text, encoding="utf-8")
    print("README.md updated")


def run_gate(args) -> None:
    main = json.loads((RESULTS / f"retrieval__{args.tag}.json").read_text(encoding="utf-8"))
    o = main["modes"]["hybrid_rerank"]["overall"]
    print(f"hybrid_rerank hit@5={o['hit@5']:.3f} (min {args.min_hit5}), mrr={o['mrr@10']:.3f} (min {args.min_mrr})")
    if o["hit@5"] < args.min_hit5 or o["mrr@10"] < args.min_mrr:
        raise SystemExit("retrieval quality regression")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--embedding-model")
    r.add_argument("--reranker-model")
    r.add_argument("--rerank-pool", type=int)
    r.add_argument("--rerank-max-chars", type=int)
    r.add_argument("--modes", help="comma-separated subset of " + ",".join(MODES))
    r.add_argument("--tag", default="default")
    g = sub.add_parser("generate")
    g.add_argument("--limit", type=int)
    sub.add_parser("report")
    gate = sub.add_parser("gate")
    gate.add_argument("--tag", default="default")
    gate.add_argument("--min-hit5", type=float, default=0.85)
    gate.add_argument("--min-mrr", type=float, default=0.6)
    args = p.parse_args()
    {"retrieval": run_retrieval, "generate": run_generate, "report": run_report, "gate": run_gate}[args.cmd](args)


if __name__ == "__main__":
    main()
