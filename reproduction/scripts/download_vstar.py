#!/usr/bin/env python3
"""Download the independent V* benchmark referenced by the official evaluator."""
import copy
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from prepare_data import ROOT, REPO, image_row


def main():
    target = ROOT / "data/vstar_bench"
    manifest = ROOT / "manifests/vstar.json"
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    info = HfApi().repo_info("craigwu/vstar_bench", repo_type="dataset", revision=old.get("revision"))
    record = {"repo_id": "craigwu/vstar_bench", "revision": info.sha, "status": "downloading"}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    snapshot_download("craigwu/vstar_bench", repo_type="dataset", revision=info.sha,
                      local_dir=target, max_workers=12,
                      allow_patterns=["README.md", "test_questions.jsonl", "direct_attributes/*", "relative_position/*"])
    raw = [json.loads(s) for s in (target / "test_questions.jsonl").read_text().splitlines() if s.strip()]
    templates = json.loads((REPO / "verl_agents/verl/utils/dataset/rl_system_prompt_template.json").read_text())
    items = [image_row({"id": r["question_id"], "question": r["text"], "answer": r["label"],
                        "image_path": r["image"], "data_source": "vstar", "ability": "visual_search",
                        "split": "test"}, target, templates) for r in raw]
    prepared = ROOT / "data/prepared"
    prepared.mkdir(exist_ok=True)
    (prepared / "vstar.json").write_text(json.dumps(items, ensure_ascii=False) + "\n")
    repeated = []
    for repeat in range(32):
        for original in items:
            item = copy.deepcopy(original)
            item["extra_info"]["original_index"] = item["extra_info"]["index"]
            item["extra_info"]["index"] += repeat * len(items)
            repeated.append(item)
    (prepared / "vstar_repeat32.json").write_text(json.dumps(repeated, ensure_ascii=False) + "\n")
    record.update(status="complete", questions=len(items), avg32_samples=len(repeated), held_out=True)
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
