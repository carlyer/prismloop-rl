#!/usr/bin/env python3
"""Check the selected local judge through the unmodified upstream reward function."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "verl_agents"))


def build_cases():
    rows = json.loads((ROOT / "data/PyVision-Image-RL-Data/pyvision_image_rl_data.json").read_text())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["data_source"]].append(row)
    antonyms = {"left": "right", "right": "left", "yes": "no", "no": "yes",
                "blue": "red", "red": "blue", "black": "white", "white": "black",
                "wood": "metal", "wooden": "metal", "metal": "wood", "above": "below",
                "below": "above", "green": "purple", "yellow": "blue", "orange": "black",
                "brown": "pink", "pink": "brown", "plastic": "metal", "glass": "wood",
                "cotton": "rubber", "rubber": "cotton", "leather": "rubber"}
    cases = []
    rng = random.Random(20261016)
    for source in sorted(grouped):
        rng.shuffle(grouped[source])
        selected = 0
        for row in grouped[source]:
            gold = str(row["answer"]).strip()
            if not gold or len(gold) > 80:
                continue
            options = set(a or b for a, b in re.findall(r"\(([A-F])\)|\b([A-F])[.、:]\s", row["question"]))
            if gold in options and len(options) >= 2:
                wrong = next(s for s in sorted(options) if s != gold)
            elif gold.lower() in antonyms:
                wrong = antonyms[gold.lower()]
            elif (re.match(r"^(yes|no)\b", gold, re.I) and
                  re.match(r"^(is|are|do|does|did|can|has|have|was|were|will|would|could|should)\b", row["question"], re.I)):
                wrong = "No" if gold.lower().startswith("yes") else "Yes"
            elif re.fullmatch(r"-?\d+(?:\.\d+)?", gold):
                number = Decimal(gold)
                wrong = str(number + max(abs(number) / 2, Decimal(1)))
            else:
                continue
            for answer, label in [(gold, 1), (wrong, 0)]:
                cases.append({"id": f"actual-{source}-{row['id']}-{label}", "source": source,
                              "question": row["question"], "gold": gold, "answer": answer,
                              "expected": label, "category": "actual_question_controlled_answer",
                              "repeats": 1, "critical": False})
            selected += 1
            if selected == 8:
                break
        if selected < 8:
            raise RuntimeError(f"Insufficient unambiguous calibration samples for {source}")
    # Independent hand-specified checks; these are NOT benchmark performance estimates.
    manual = [
        ("How many objects are shown?", "2", "There are two objects.", 1, "paraphrase"),
        ("How many objects are shown?", "2", "There are three objects.", 0, "count_error"),
        ("What color is the bag?", "blue", "The bag is blue.", 1, "paraphrase"),
        ("What color is the bag?", "blue", "The bag is red.", 0, "attribute_error"),
        ("Is the cup left or right of the book?", "left", "It is to the left of the book.", 1, "paraphrase"),
        ("Is the cup left or right of the book?", "left", "It is to the right of the book.", 0, "direction_error"),
        ("What is 1 divided by 2?", "1/2", "0.5", 1, "numeric_equivalence"),
        ("What is 1 divided by 2?", "1/2", "0.05", 0, "decimal_error"),
        ("Express one quarter as a percentage.", "25%", "0.25", 1, "numeric_equivalence"),
        ("Express one quarter as a percentage.", "25%", "2.5%", 0, "percentage_error"),
        ("How long is the segment?", "25 cm", "0.25 m", 1, "unit_equivalence"),
        ("How long is the segment?", "25 cm", "0.25 cm", 0, "unit_error"),
        ("Which shape is shown? (A) triangle (B) square (C) circle", "B", "square", 1, "option_equivalence"),
        ("Which shape is shown? (A) triangle (B) square (C) circle", "B", "A", 0, "option_error"),
        ("Which shape is shown? (A) triangle (B) square (C) circle", "B", "B. square", 1, "option_equivalence"),
        ("Which shape is shown? (A) triangle (B) square (C) circle", "B", "C. circle", 0, "option_error"),
        ("What is the value of x?", "-3", "negative three", 1, "numeric_equivalence"),
        ("What is the value of x?", "-3", "3", 0, "sign_error"),
        ("Give the simplified value of sqrt(8).", "2*sqrt(2)", "sqrt(8)", 1, "symbolic_equivalence"),
        ("Give the simplified value of sqrt(8).", "2*sqrt(2)", "sqrt(2)", 0, "symbolic_error"),
        ("图中有几个杯子？", "3", "一共有三个杯子。", 1, "chinese_paraphrase"),
        ("图中有几个杯子？", "3", "一共有四个杯子。", 0, "chinese_error"),
        ("Is the lamp switched on?", "yes", "Yes, the lamp is on.", 1, "paraphrase"),
        ("Is the lamp switched on?", "yes", "No, it is switched off.", 0, "negation_error"),
        ("What is the material of the glove?", "rubber", "It is made of rubber.", 1, "paraphrase"),
        ("What is the material of the glove?", "rubber", "It is made of cotton.", 0, "material_error"),
        ("What is the exact integer result of 6 times 7?", "42", "42", 1, "exact_match"),
        ("What is the exact integer result of 6 times 7?", "42", "41", 0, "near_numeric_error"),
        ("What is the height?", "2 m", "200 cm", 1, "unit_equivalence"),
        ("What is the height?", "2 m", "200 m", 0, "unit_error"),
        ("Which category is shown? A. cat B. dog", "B", "dog", 1, "option_equivalence"),
        ("Which category is shown? A. cat B. dog", "B", "cat", 0, "option_error"),
    ]
    for i, (question, gold, answer, label, category) in enumerate(manual):
        cases.append({"id": f"manual-{i:02}", "source": "wemath-standard" if any(s in category for s in ["numeric", "unit", "symbolic", "percentage", "decimal", "sign", "option", "exact"]) else "deepeyes",
                      "question": question, "gold": gold, "answer": answer,
                      "expected": label, "category": category, "repeats": 3, "critical": label == 0})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    from openai import OpenAI
    from verl.utils.reward_score import multi_task_verifier as verifier
    config = json.loads((ROOT / "judge-config.json").read_text())
    selection = json.loads((ROOT / "judge-selection.json").read_text())
    cases = build_cases()
    case_path = ROOT / "manifests/judge-calibration-cases.json"
    case_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
    results_path = ROOT / "logs/judge-calibration-results.jsonl"
    # Establish gates before running inference.
    gates = {"minimum_accuracy": 0.95, "maximum_false_positive_rate": 0.02,
             "minimum_valid_output_rate": 1.0, "maximum_critical_errors": 0}
    (ROOT / "manifests/judge-calibration-plan.json").write_text(json.dumps({
        "unique_cases": len(cases), "trials": sum(c["repeats"] for c in cases),
        "gates": gates, "temperature": 0.3, "max_tokens": 32,
        "judge_selection": selection, "upstream_reward_code_unmodified": True,
        "purpose": "Engineering calibration; not proof of equivalence to the paper's 72B judge."
    }, ensure_ascii=False, indent=2) + "\n")
    real_client = OpenAI(api_key=config["api_key"], base_url=config["base_url"], timeout=120, max_retries=1)
    local = threading.local()
    def client_factory(**kwargs):
        def create(**request):
            request.update(seed=local.seed, max_tokens=32)
            response = real_client.chat.completions.create(**request)
            local.raw = response.choices[0].message.content or ""
            local.usage = response.usage.model_dump() if response.usage else {}
            return response
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    tasks = [(c, repeat) for c in cases for repeat in range(c["repeats"])]
    def run(task):
        case, repeat = task
        local.seed = 20261016 + int(hashlib.sha256((case["id"] + str(repeat)).encode()).hexdigest()[:6], 16)
        local.raw = None
        local.usage = {}
        start = time.monotonic()
        try:
            answer = "<answer>\n\\boxed{" + case["answer"] + "}\n</answer>"
            score = verifier.compute_score(answer, case["gold"], {"question": case["question"]}, config, case["source"])
            prediction = int(score["score"])
            valid = bool(local.raw is not None and re.fullmatch(r"(?:Judgement:\s*)?[01]", local.raw.strip()))
            return {**case, "repeat": repeat, "seed": local.seed, "prediction": prediction,
                    "matches_expected": prediction == case["expected"], "valid_output": valid,
                    "raw_judge_output": local.raw, "usage": local.usage,
                    "latency_seconds": round(time.monotonic() - start, 3)}
        except Exception as exc:
            return {**case, "repeat": repeat, "prediction": None, "matches_expected": False,
                    "valid_output": False, "error": str(exc)}
    results = []
    print(f"Running {len(tasks)} trials over {len(cases)} unique cases", flush=True)
    with patch.object(verifier, "OpenAI", client_factory), ThreadPoolExecutor(max_workers=args.workers) as pool, results_path.open("w") as log:
        for result in pool.map(run, tasks):
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if not result["matches_expected"]:
                print("MISMATCH", result["id"], result.get("raw_judge_output"), result.get("error"), flush=True)
            if len(results) % 20 == 0:
                print(f"Completed {len(results)}/{len(tasks)}", flush=True)
    total = len(results)
    negatives = [r for r in results if r["expected"] == 0]
    false_positives = sum(r["prediction"] == 1 for r in negatives)
    correct = sum(r["matches_expected"] for r in results)
    valid = sum(r["valid_output"] for r in results)
    critical_errors = sum(r["critical"] and not r["matches_expected"] for r in results)
    accuracy = correct / total
    fpr = false_positives / len(negatives)
    passed = (accuracy >= gates["minimum_accuracy"] and fpr <= gates["maximum_false_positive_rate"]
              and valid / total >= gates["minimum_valid_output_rate"] and critical_errors == 0)
    errors = [r for r in results if not r["matches_expected"]]
    report = {"status": "passed" if passed else "failed", "judge_selection": selection,
              "unique_cases": len(cases), "trials": total, "correct": correct,
              "accuracy": accuracy, "false_positive_rate": fpr, "valid_output_rate": valid / total,
              "critical_errors": critical_errors, "gates": gates,
              "categories": dict(Counter(c["category"] for c in cases)), "errors": errors,
              "case_file_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
              "upstream_verifier_sha256": hashlib.sha256(Path(verifier.__file__).read_bytes()).hexdigest(),
              "selection_sha256": hashlib.sha256((ROOT / "judge-selection.json").read_bytes()).hexdigest(),
              "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "limitation": "Limited controlled calibration; no agreement measurement against the original 72B judge."}
    (ROOT / "manifests/judge-calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["errors", "categories"]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
