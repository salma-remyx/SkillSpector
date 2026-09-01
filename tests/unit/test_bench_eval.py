# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from scripts import bench_eval, compare_scan_accuracy

CATEGORY_RULES = {"prompt_injection": {"PI": {"min": 1}}}


def _complete_report(issues: list[dict[str, object]]) -> dict[str, object]:
    return {
        "issues": issues,
        "execution_successful": True,
        "analysis_completeness": {
            "total_components": 1,
            "scanned_components": 1,
            "coverage_percent": 100.0,
            "is_complete": True,
            "status": "complete",
            "execution_successful": True,
            "fully_inspected_files": 1,
            "partially_inspected_files": 0,
            "entirely_uninspected_files": 0,
            "ledger_exceptions": [],
            "scope_exclusions": [],
            "analyzer_statuses": [],
            "limitations": [],
        },
    }


def _write_records(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _benchmark_records() -> list[dict[str, Any]]:
    """Five records spanning two attack categories and two sources."""
    return [
        {
            "id": "m-pi-1",
            "label": "malicious",
            "attack_category": "prompt_injection",
            "source": "alpha",
            "skill_text": "# injected instructions\n",
        },
        {
            "id": "m-pi-2",
            "label": "malicious",
            "attack_category": "prompt_injection",
            "source": "beta",
            "skill_text": "# injected instructions\n",
        },
        {
            "id": "m-ch-1",
            "label": "malicious",
            "attack_category": "credential_theft",
            "source": "alpha",
            "files": {"scripts/harvest.sh": "cat ~/.ssh/id_rsa\n"},
            "expected_rules": {"CRED": 1},
        },
        {"id": "b-1", "label": "benign", "source": "alpha", "skill_text": "# safe\n"},
        {"id": "b-2", "label": "benign", "source": "beta", "skill_text": "# safe\n"},
    ]


def _write_reports(root: Path, plan: dict[str, list[dict[str, object]]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for record_id, issues in plan.items():
        (root / f"{record_id}.json").write_text(
            json.dumps(_complete_report(issues)),
            encoding="utf-8",
        )
    return root


def test_materialize_emits_an_accuracy_gate_manifest(tmp_path: Path) -> None:
    records = _write_records(tmp_path / "bench.jsonl", _benchmark_records())
    corpus_root = tmp_path / "corpus"
    manifest_path = tmp_path / "manifest.json"

    summary = bench_eval.materialize_benchmark(
        records_path=records,
        corpus_root=corpus_root,
        manifest_path=manifest_path,
        category_rules=CATEGORY_RULES,
    )

    assert summary["records"] == 5
    assert summary["malicious"] == 3
    assert summary["benign"] == 2
    assert summary["attack_categories"] == ["credential_theft", "prompt_injection"]
    assert summary["sources"] == ["alpha", "beta"]
    assert (corpus_root / "m-pi-1" / "SKILL.md").is_file()
    assert (corpus_root / "m-ch-1" / "scripts" / "harvest.sh").is_file()

    manifest, _ = compare_scan_accuracy._load_manifest(manifest_path)
    assert manifest["schema_version"] == compare_scan_accuracy.SCHEMA_VERSION
    assert manifest["material_regression_policy"] == {
        field: 0 for field in compare_scan_accuracy.POLICY_FIELDS
    }
    cases = {case["id"]: case for case in manifest["cases"]}
    assert {case["classification"] for case in cases.values()} == set(
        compare_scan_accuracy.REQUIRED_CLASSIFICATIONS
    )
    # A category default pins expectations when the record itself does not.
    assert cases["m-pi-1"]["expected_rules"] == {"PI": {"min": 1}}
    # An explicit record-level pin wins over the category default.
    assert cases["m-ch-1"]["expected_rules"] == {"CRED": 1}
    # Benign corpus entries stay zero-tolerance.
    assert cases["b-1"]["expected_rules"] == {}

    # Everything the accuracy gate itself validates before running a scan.
    compare_scan_accuracy._validate_policy(manifest)
    for case in manifest["cases"]:
        compare_scan_accuracy._adjudicate_counts(case, Counter(), frozenset())
        assert compare_scan_accuracy._resolve_case_path(corpus_root, case["path"]).is_dir()
    # Pinned expectations drive the gate's cohort accounting in both directions.
    missed = compare_scan_accuracy._adjudicate_counts(cases["m-pi-1"], Counter(), frozenset())
    assert missed["false_negatives"] == 1
    over_fired = compare_scan_accuracy._adjudicate_counts(
        cases["m-pi-1"], Counter({"PI": 2}), frozenset()
    )
    assert over_fired["false_positives"] == 1


def test_materialize_rejects_ambiguous_records(tmp_path: Path) -> None:
    def materialize(rows: list[dict[str, Any]]) -> None:
        bench_eval.materialize_benchmark(
            records_path=_write_records(tmp_path / "bench.jsonl", rows),
            corpus_root=tmp_path / "corpus",
            manifest_path=tmp_path / "manifest.json",
            category_rules=CATEGORY_RULES,
        )

    with pytest.raises(ValueError, match="pins no expected_rules"):
        materialize(
            [
                {
                    "id": "m-1",
                    "label": "malicious",
                    "attack_category": "credential_theft",
                    "skill_text": "# x\n",
                },
            ]
        )
    with pytest.raises(ValueError, match="must pin no expected_rules"):
        materialize(
            [
                {
                    "id": "b-1",
                    "label": "benign",
                    "skill_text": "# x\n",
                    "expected_rules": {"PI": 1},
                },
            ]
        )
    with pytest.raises(ValueError, match="must not carry an attack_category"):
        materialize(
            [{"id": "b-1", "label": "benign", "attack_category": "x", "skill_text": "# x\n"}],
        )
    with pytest.raises(ValueError, match="duplicates benchmark id"):
        materialize(
            [
                {"id": "b-1", "label": "benign", "skill_text": "# x\n"},
                {"id": "b-1", "label": "benign", "skill_text": "# x\n"},
            ]
        )
    with pytest.raises(ValueError, match="no path separators"):
        materialize([{"id": "nested/id", "label": "benign", "skill_text": "# x\n"}])
    with pytest.raises(ValueError, match="safe relative paths"):
        materialize([{"id": "b-1", "label": "benign", "files": {"../escape.sh": "x"}}])
    with pytest.raises(ValueError, match="unknown field"):
        materialize([{"id": "b-1", "label": "benign", "skill_text": "# x\n", "extra": 1}])


def test_materialize_refuses_a_dirty_corpus_root(tmp_path: Path) -> None:
    records = _write_records(tmp_path / "bench.jsonl", _benchmark_records())
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "leftover").mkdir()
    with pytest.raises(ValueError, match="not empty"):
        bench_eval.materialize_benchmark(
            records_path=records,
            corpus_root=corpus_root,
            manifest_path=tmp_path / "manifest.json",
            category_rules=CATEGORY_RULES,
        )


def test_score_reports_joint_detection_and_source_disjoint_metrics(tmp_path: Path) -> None:
    records = _write_records(tmp_path / "bench.jsonl", _benchmark_records())
    reports = _write_reports(
        tmp_path / "reports",
        {
            "m-pi-1": [{"id": "PI"}, {"id": "PI"}],
            "m-pi-2": [],
            "m-ch-1": [{"id": "CRED"}],
            "b-1": [],
            "b-2": [{"id": "X"}],
        },
    )

    result = bench_eval.score_benchmark(records_path=records, reports_root=reports)

    assert result["records"] == 5
    # tp=2, fn=1, fp=1, tn=1 over the whole (random-split) benchmark.
    assert result["random"]["flagged_malicious"] == 2
    assert result["random"]["unflagged_malicious"] == 1
    assert result["random"]["flagged_benign"] == 1
    assert result["random"]["unflagged_benign"] == 1
    assert result["random"]["malicious_recall"] == pytest.approx(2 / 3)
    assert result["random"]["benign_false_positive_rate"] == pytest.approx(0.5)
    assert result["random"]["f1_malicious"] == pytest.approx(2 / 3)
    assert result["random"]["f1_benign"] == pytest.approx(0.5)
    assert result["random"]["macro_f1"] == pytest.approx(7 / 12)

    # Source-disjoint folds: perfect on alpha, and on beta the missed malicious
    # skill coincides with the over-flagged benign one.
    assert result["source_disjoint"]["alpha"]["malicious_recall"] == pytest.approx(1.0)
    assert result["source_disjoint"]["alpha"]["benign_false_positive_rate"] == pytest.approx(0.0)
    assert result["source_disjoint"]["alpha"]["macro_f1"] == pytest.approx(1.0)
    assert result["source_disjoint"]["beta"]["malicious_recall"] == pytest.approx(0.0)
    assert result["source_disjoint"]["beta"]["benign_false_positive_rate"] == pytest.approx(1.0)
    assert result["source_disjoint"]["beta"]["macro_f1"] == pytest.approx(0.0)

    assert result["by_attack_category"]["prompt_injection"] == {
        "records": 2,
        "flagged": 1,
        "malicious_recall": pytest.approx(0.5),
    }
    credential_theft = result["by_attack_category"]["credential_theft"]
    assert credential_theft["malicious_recall"] == pytest.approx(1.0)

    # Raising the flag threshold trades recall for benign precision.
    stricter = bench_eval.score_benchmark(
        records_path=records,
        reports_root=reports,
        threshold=2,
    )
    assert stricter["random"]["flagged_malicious"] == 1
    assert stricter["random"]["flagged_benign"] == 0
    assert stricter["random"]["malicious_recall"] == pytest.approx(1 / 3)
    assert stricter["random"]["benign_false_positive_rate"] == pytest.approx(0.0)


def test_score_rejects_missing_and_incomplete_reports(tmp_path: Path) -> None:
    records = _write_records(tmp_path / "bench.jsonl", _benchmark_records())
    reports = _write_reports(
        tmp_path / "reports",
        {
            "m-pi-1": [],
            "m-pi-2": [],
            "m-ch-1": [],
            "b-1": [],
        },
    )
    with pytest.raises(ValueError, match="Missing scanner report"):
        bench_eval.score_benchmark(records_path=records, reports_root=reports)

    (reports / "b-2.json").write_text(
        json.dumps({"issues": [], "execution_successful": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not execution-successful"):
        bench_eval.score_benchmark(records_path=records, reports_root=reports)


def test_scan_cases_runs_one_isolated_cli_scan_per_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    (corpus_root / "case-a").mkdir(parents=True)
    (corpus_root / "case-b").mkdir(parents=True)
    (corpus_root / "stray.txt").write_text("ignored", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        commands.append(command)
        target = Path(command[4])
        issues: list[dict[str, object]] = [] if target.name == "case-a" else [{"id": "R1"}]
        return subprocess.CompletedProcess(
            command,
            1,
            json.dumps(_complete_report(issues)).encode("utf-8"),
            b"",
        )

    monkeypatch.setattr(bench_eval.subprocess, "run", fake_run)
    result = bench_eval.scan_cases(corpus_root=corpus_root, reports_root=tmp_path / "reports")

    assert result["scanned"] == 2
    assert commands == [
        [
            sys.executable,
            "-m",
            "skillspector.cli",
            "scan",
            str(corpus_root / "case-a"),
            "--format",
            "json",
            "--no-llm",
        ],
        [
            sys.executable,
            "-m",
            "skillspector.cli",
            "scan",
            str(corpus_root / "case-b"),
            "--format",
            "json",
            "--no-llm",
        ],
    ]
    stored = json.loads((tmp_path / "reports" / "case-a.json").read_text(encoding="utf-8"))
    assert stored["issues"] == []


def test_main_reports_errors_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    records = _write_records(tmp_path / "bench.jsonl", _benchmark_records())
    exit_code = bench_eval.main(
        [
            "materialize",
            "--records",
            str(records),
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )
    assert exit_code == 2
    assert "pins no expected_rules" in capsys.readouterr().err
