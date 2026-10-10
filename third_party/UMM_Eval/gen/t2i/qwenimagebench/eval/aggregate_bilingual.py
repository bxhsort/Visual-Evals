#!/usr/bin/env python3
"""Merge EN + ZH judge bench_scores into one summary.

Each input is a judge.py *_bench_scores.json:
    {"level1": {<dim>: score|null, ...}, "level2": {...}, "total": score|null}
Output: per-language blocks + an "overall" mean across the languages present
(per-L1-pillar mean and total mean).
"""
import argparse
import json
import os


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", help="EN *_bench_scores.json", default=None)
    ap.add_argument("--zh", help="ZH *_bench_scores.json", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    langs = {}
    for lang, path in (("en", args.en), ("zh", args.zh)):
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                langs[lang] = json.load(f)

    if not langs:
        raise SystemExit("ERROR: no per-language bench_scores found to aggregate.")

    all_l1 = sorted({d for b in langs.values() for d in b.get("level1", {})})
    overall = {
        "level1": {d: _mean([b.get("level1", {}).get(d) for b in langs.values()]) for d in all_l1},
        "total": _mean([b.get("total") for b in langs.values()]),
    }
    result = {"per_language": langs, "overall": overall, "languages": sorted(langs)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("QWEN-IMAGE-BENCH — bilingual summary")
    for lang, b in langs.items():
        t = b.get("total")
        print(f"  [{lang}] total: {t:.2f}" if t is not None else f"  [{lang}] total: N/A")
    ot = overall["total"]
    print(f"  [overall] total: {ot:.2f}" if ot is not None else "  [overall] total: N/A")
    print(f"  -> {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
