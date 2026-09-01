# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate SkillSpector on a labeled malicious/benign Agent-Skill benchmark.

Implements the evaluation protocol of MaliciousSkillBench (arXiv:2608.19901):
detection quality is reported as the JOINT pair of malicious recall and benign
false-positive rate — with macro-F1, per-attack-category recall, and per-source
("source-disjoint") folds — instead of a single accuracy number. The benchmark
itself stays an external input, exactly like the accuracy gate's adjudicated
corpus.

Three stages:

  materialize  benchmark records (JSONL) -> corpus directories plus an
               accuracy-gate manifest consumable by compare_scan_accuracy.py
  scan         run the installed SkillSpector CLI once per corpus case and
               store one ``--format json --no-llm`` report per record
  score        per-record reports -> the benchmark metric document

SkillSpector is a static scanner, so nothing is ever trained on in-split
sources; its per-source scores therefore coincide with the benchmark's
source-disjoint operating point, and the score document reports those folds
under that name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

BENCH_SCHEMA_VERSION = 1
RECORD_FIELDS = frozenset(
    {"id", "label", "attack_category", "source", "skill_text", "files", "expected_rules"}
)
LABELS = frozenset({"malicious", "benign"})
# The benchmark label set maps onto the accuracy gate's adjudication classes:
# benign skills are zero-tolerance maintained corpora, malicious skills are
# real-world cases whose expected detections a reviewer has pinned.
BENIGN_CLASSIFICATION = "maintained_benign"
MALICIOUS_CLASSIFICATION = "approved_real_world"
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SCAN_ARGUMENTS = ("--format", "json", "--no-llm")
_SCAN_TIMEOUT_SECONDS = 600.0


def _load_accuracy_gate() -> Any:
    """Import the sibling accuracy-gate module that owns the manifest contract.

    The gate lives outside the installed package, so it is loaded from its
    file path. Its validation helpers are deliberately reused — the same ones
    ``compare_scanners`` applies before any scan runs — so a manifest emitted
    here cannot drift from what the accuracy gate accepts.
    """
    module_path = Path(__file__).resolve().parent / "compare_scan_accuracy.py"
    spec = importlib.util.spec_from_file_location("compare_scan_accuracy", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - a .py file always has a loader
        raise RuntimeError(f"Cannot import the accuracy gate from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ACCURACY_GATE = _load_accuracy_gate()


def _zero_tolerance_policy() -> dict[str, int]:
    return {field: 0 for field in _ACCURACY_GATE.POLICY_FIELDS}


def _validated_record(
    raw: dict[str, Any],
    where: str,
    seen_ids: set[str],
) -> dict[str, Any]:
    unknown_fields = sorted(set(raw) - RECORD_FIELDS)
    if unknown_fields:
        raise ValueError(f"{where} has unknown field(s): " + ", ".join(unknown_fields))
    record_id = raw.get("id")
    if not isinstance(record_id, str) or not _ID_PATTERN.fullmatch(record_id):
        raise ValueError(
            f"{where} needs an id of letters, digits, '.', '_' or '-' with no path separators"
        )
    if record_id in seen_ids:
        raise ValueError(f"{where} duplicates benchmark id: {record_id}")
    seen_ids.add(record_id)
    label = raw.get("label")
    if label not in LABELS:
        raise ValueError(f"{where} label must be one of: " + ", ".join(sorted(LABELS)))
    category = raw.get("attack_category")
    if label == "malicious":
        if not isinstance(category, str) or not category.strip() or category != category.strip():
            raise ValueError(f"{where} malicious record {record_id} needs an attack_category")
    elif category not in (None, ""):
        raise ValueError(f"{where} benign record {record_id} must not carry an attack_category")
    source = raw.get("source")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError(f"{where} source must be a non-empty string when present")
    skill_text = raw.get("skill_text")
    if skill_text is not None and not isinstance(skill_text, str):
        raise ValueError(f"{where} skill_text must be a string")
    files = raw.get("files")
    if files is not None:
        if not isinstance(files, dict) or not files:
            raise ValueError(f"{where} files must be a non-empty object")
        for name, content in files.items():
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError(f"{where} file entries must map names to text content")
            candidate = Path(name)
            if not name.strip() or candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(
                    f"{where} file entries must be safe relative paths with text content"
                )
        if skill_text is not None and "SKILL.md" in files:
            raise ValueError(f"{where} record {record_id} defines SKILL.md twice")
    if skill_text is None and not files:
        raise ValueError(f"{where} record {record_id} needs skill_text and/or files")
    expected_rules = raw.get("expected_rules")
    if expected_rules is not None and not isinstance(expected_rules, dict):
        raise ValueError(f"{where} expected_rules must be an object")
    if label == "benign" and expected_rules:
        raise ValueError(f"{where} benign record {record_id} must pin no expected_rules")
    return {
        "id": record_id,
        "label": label,
        "attack_category": category if label == "malicious" else None,
        "source": source,
        "skill_text": skill_text,
        "files": files,
        "expected_rules": dict(expected_rules) if expected_rules else {},
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    """Parse and fully validate every benchmark record before writing anything."""
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where} is not valid JSON: {error}") from error
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a JSON object")
        records.append(_validated_record(raw, where, seen_ids))
    if not records:
        raise ValueError(f"{path} contains no benchmark records")
    return records


def _expected_rules_for(
    record: dict[str, Any],
    category_rules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve the adjudicated expectations for one record.

    Malicious records need pinned expectations — from the record itself or
    from its attack-category default — because a zero-expectation malicious
    entry would make the accuracy gate score every true detection on that
    record as a false positive.
    """
    if record["label"] == "benign":
        return {}
    if record["expected_rules"]:
        return dict(record["expected_rules"])
    category = str(record["attack_category"])
    pinned = category_rules.get(category)
    if pinned:
        return dict(pinned)
    raise ValueError(
        f"Malicious record {record['id']} (attack_category {category}) pins no "
        "expected_rules and no category default exists; refusing to emit a "
        "zero-expectation malicious manifest entry"
    )


def _write_case(case_root: Path, record: dict[str, Any]) -> None:
    case_root.mkdir(parents=True, exist_ok=False)
    if record["skill_text"] is not None:
        (case_root / "SKILL.md").write_text(record["skill_text"], encoding="utf-8")
    for name, content in sorted((record["files"] or {}).items()):
        target = case_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _validate_against_gate(
    manifest: dict[str, Any],
    corpus_root: Path,
) -> None:
    """Re-run the accuracy gate's own manifest validation on what we emit."""
    _ACCURACY_GATE._validate_policy(manifest)
    for case in manifest["cases"]:
        # The exact pre-scan adjudication call compare_scanners makes per case.
        _ACCURACY_GATE._adjudicate_counts(case, Counter(), frozenset())
        _ACCURACY_GATE._resolve_case_path(corpus_root, str(case["path"]))


def materialize_benchmark(
    *,
    records_path: Path,
    corpus_root: Path,
    manifest_path: Path,
    category_rules: dict[str, dict[str, Any]] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the corpus directories and accuracy-gate manifest for the records.

    Materialize into a fresh corpus root: validation failures can leave a
    partially written corpus behind, never a partially written manifest.
    """
    if {BENIGN_CLASSIFICATION, MALICIOUS_CLASSIFICATION} != set(
        _ACCURACY_GATE.REQUIRED_CLASSIFICATIONS
    ):
        raise ValueError("Accuracy gate classifications changed; update the benchmark mapping")
    records = _load_records(records_path)
    rules = category_rules or {}
    cases = [
        {
            "id": record["id"],
            "path": record["id"],
            "classification": (
                MALICIOUS_CLASSIFICATION
                if record["label"] == "malicious"
                else BENIGN_CLASSIFICATION
            ),
            "expected_rules": _expected_rules_for(record, rules),
        }
        for record in records
    ]
    manifest = {
        "schema_version": _ACCURACY_GATE.SCHEMA_VERSION,
        "material_regression_policy": policy or _zero_tolerance_policy(),
        "cases": cases,
    }
    if corpus_root.exists() and any(corpus_root.iterdir()):
        raise ValueError(f"Corpus root is not empty: {corpus_root}")
    for record in records:
        _write_case(corpus_root / record["id"], record)
    _validate_against_gate(manifest, corpus_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": BENCH_SCHEMA_VERSION,
        "records": len(records),
        "malicious": sum(1 for record in records if record["label"] == "malicious"),
        "benign": sum(1 for record in records if record["label"] == "benign"),
        "attack_categories": sorted(
            {str(record["attack_category"]) for record in records if record["attack_category"]}
        ),
        "sources": sorted({str(record["source"]) for record in records if record["source"]}),
        "corpus_root": str(corpus_root),
        "manifest": str(manifest_path),
    }


def scan_cases(*, corpus_root: Path, reports_root: Path) -> dict[str, Any]:
    """Scan every corpus case with the installed SkillSpector CLI.

    Each case runs in its own interpreter — mirroring the isolated per-case
    scans the accuracy gate performs — and writes ``<case-id>.json`` holding
    the report that ``score`` consumes.
    """
    if not corpus_root.is_dir():
        raise ValueError(f"Corpus root is not a directory: {corpus_root}")
    case_dirs = sorted(path for path in corpus_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"Corpus root has no case directories: {corpus_root}")
    reports_root.mkdir(parents=True, exist_ok=True)
    scanned: list[dict[str, Any]] = []
    for case_dir in case_dirs:
        completed = subprocess.run(
            [sys.executable, "-m", "skillspector.cli", "scan", str(case_dir), *_SCAN_ARGUMENTS],
            capture_output=True,
            timeout=_SCAN_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Scanner exited {completed.returncode} for {case_dir.name}: {detail}"
            )
        report_path = reports_root / f"{case_dir.name}.json"
        report_path.write_bytes(completed.stdout)
        scanned.append(
            {
                "id": case_dir.name,
                "report": str(report_path),
                "exit_code": completed.returncode,
            }
        )
    return {
        "schema_version": BENCH_SCHEMA_VERSION,
        "scanned": len(scanned),
        "cases": scanned,
    }


def _load_report(path: Path) -> dict[str, Any]:
    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_ACCURACY_GATE._reject_duplicate_json_pairs,
    )
    if not isinstance(document, dict):
        raise ValueError(f"Scanner report is not a JSON object: {path}")
    return document


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _confusion(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for verdict in verdicts:
        if verdict["label"] == "malicious":
            counts["tp" if verdict["flagged"] else "fn"] += 1
        else:
            counts["fp" if verdict["flagged"] else "tn"] += 1
    return counts


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    malicious_recall = _ratio(tp, tp + fn)
    benign_fpr = _ratio(fp, fp + tn)
    precision = _ratio(tp, tp + fp)
    f1_malicious = _f1(precision, malicious_recall)
    f1_benign = _f1(_ratio(tn, tn + fn), _ratio(tn, tn + fp))
    return {
        "flagged_malicious": tp,
        "unflagged_malicious": fn,
        "flagged_benign": fp,
        "unflagged_benign": tn,
        "malicious_recall": malicious_recall,
        "benign_false_positive_rate": benign_fpr,
        "precision": precision,
        "f1_malicious": f1_malicious,
        "f1_benign": f1_benign,
        "macro_f1": (f1_malicious + f1_benign) / 2,
    }


def score_benchmark(
    *,
    records_path: Path,
    reports_root: Path,
    threshold: int = 1,
    selected_rules: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Score per-record scanner reports under the joint detection protocol."""
    if threshold < 1:
        raise ValueError("Flag threshold must be at least one occurrence")
    records = _load_records(records_path)
    missing = [
        record["id"]
        for record in records
        if not (reports_root / f"{record['id']}.json").is_file()
    ]
    if missing:
        raise ValueError("Missing scanner report(s): " + ", ".join(missing))
    verdicts: list[dict[str, Any]] = []
    for record in records:
        report = _load_report(reports_root / f"{record['id']}.json")
        # The gate's report reader rejects failed, partial, or ambiguous scans.
        counts = _ACCURACY_GATE._rule_counts(report, selected_rules)
        occurrences = sum(counts.values())
        verdicts.append(
            {
                "id": record["id"],
                "label": record["label"],
                "attack_category": record["attack_category"],
                "source": record["source"],
                "flagged": occurrences >= threshold,
                "occurrences": occurrences,
                "by_rule": dict(sorted(counts.items())),
            }
        )
    by_attack_category: dict[str, dict[str, Any]] = {}
    for category in sorted(
        {str(verdict["attack_category"]) for verdict in verdicts if verdict["attack_category"]}
    ):
        subset = [verdict for verdict in verdicts if verdict["attack_category"] == category]
        flagged = sum(1 for verdict in subset if verdict["flagged"])
        by_attack_category[category] = {
            "records": len(subset),
            "flagged": flagged,
            "malicious_recall": _ratio(flagged, len(subset)),
        }
    sources = sorted({str(verdict["source"]) for verdict in verdicts if verdict["source"]})
    return {
        "schema_version": BENCH_SCHEMA_VERSION,
        "decision_rule": {
            "flagged_when_occurrences_at_least": threshold,
            "selected_rules": sorted(selected_rules),
            "count_unit": "occurrence",
        },
        "records": len(verdicts),
        "records_without_source": sum(1 for verdict in verdicts if not verdict["source"]),
        "random": _metrics(_confusion(verdicts)),
        "source_disjoint": {
            source: _metrics(
                _confusion([verdict for verdict in verdicts if verdict["source"] == source])
            )
            for source in sources
        },
        "by_attack_category": by_attack_category,
        "verdicts": verdicts,
    }


def _load_sidecar(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} document must be a JSON object: {path}")
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser(
        "materialize",
        help="write a corpus plus accuracy-gate manifest from benchmark records",
    )
    materialize.add_argument("--records", type=Path, required=True)
    materialize.add_argument("--corpus-root", type=Path, required=True)
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument(
        "--category-rules",
        type=Path,
        help="JSON mapping attack_category to default expected_rules",
    )
    materialize.add_argument(
        "--policy",
        type=Path,
        help="JSON material_regression_policy override (default: zero tolerance)",
    )

    scan = subparsers.add_parser(
        "scan",
        help="scan every corpus case with the installed SkillSpector CLI",
    )
    scan.add_argument("--corpus-root", type=Path, required=True)
    scan.add_argument("--reports", type=Path, required=True)

    score = subparsers.add_parser(
        "score",
        help="compute benchmark metrics from per-record scanner reports",
    )
    score.add_argument("--records", type=Path, required=True)
    score.add_argument("--reports", type=Path, required=True)
    score.add_argument("--threshold", type=int, default=1)
    score.add_argument("--rules", help="comma-separated rule ids restricting the decision rule")
    score.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "materialize":
            result = materialize_benchmark(
                records_path=args.records,
                corpus_root=args.corpus_root,
                manifest_path=args.manifest,
                category_rules=_load_sidecar(args.category_rules, "category rules"),
                policy=_load_sidecar(args.policy, "policy"),
            )
        elif args.command == "scan":
            result = scan_cases(corpus_root=args.corpus_root, reports_root=args.reports)
        else:
            result = score_benchmark(
                records_path=args.records,
                reports_root=args.reports,
                threshold=args.threshold,
                selected_rules=(
                    frozenset(rule.strip() for rule in args.rules.split(",") if rule.strip())
                    if args.rules
                    else frozenset()
                ),
            )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"benchmark evaluation error: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if getattr(args, "output", None):
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
