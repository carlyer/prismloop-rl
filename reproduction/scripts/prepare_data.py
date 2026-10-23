#!/usr/bin/env python3
"""Extract official media and create local-path JSON without modifying originals."""
import argparse
from collections import Counter
from collections import defaultdict
import copy
import json
from pathlib import Path
import tarfile
import random

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def extract(folder, archive_name):
    archive = folder / archive_name
    marker = folder / (archive_name + ".extracted.json")
    if marker.exists():
        return
    count = 0
    print(f"Extracting {archive}", flush=True)
    # Streaming mode avoids retaining tens of thousands of TarInfo entries.
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not (member.isfile() or member.isdir()):
                continue
            target = (folder / member.name).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extract(member, path=folder, filter="data")
            count += member.isfile()
            if count % 5000 == 0 and member.isfile():
                print(f"Extracted {count} files", flush=True)
    marker.write_text(json.dumps({"archive": str(archive), "file_count": count}, indent=2))


def locate(folder, raw):
    path = Path(raw)
    choices = [folder / path, folder / "images" / path, folder / "videos" / path,
               folder / "pyvision_image_rl_data" / path]
    if path.parts and path.parts[0] in {"images", "videos"}:
        choices.append(folder.joinpath(*path.parts[1:]))
    for candidate in choices:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Media missing: {raw} (under {folder})")


def image_row(row, folder, templates):
    from PIL import Image
    image = locate(folder, row["image_path"])
    with Image.open(image) as im:
        width, height = im.size
    question = row["question"]
    content = "<image>\n" + templates["vistool_with_img_info_v2"].format(
        query=question, width=width, height=height)
    return {
        "prompt": [{"role": "user", "content": content}],
        "data_source": row["data_source"], "ability": row["ability"],
        "env_name": "pyvision_gym_w_image_hint",
        "reward_model": {"ground_truth": row["answer"], "style": "model"},
        "extra_info": {"answer": row["answer"], "index": int(row["id"]),
                       "question": question, "split": row.get("split", "train")},
        "mm_hint": {"hint_path": image, "hint_type": "image"},
    }


def prepare(kind):
    folder = ROOT / "data" / f"PyVision-{kind.capitalize()}-RL-Data"
    extract(folder, "images.tar.gz" if kind == "image" else "videos.tar.gz")
    original = folder / f"pyvision_{kind}_rl_data.json"
    rows = json.loads(original.read_text())
    templates = json.loads((REPO / "verl_agents/verl/utils/dataset/rl_system_prompt_template.json").read_text())
    output = []
    for row in rows:
        row = copy.deepcopy(row)
        if "image_path" in row:
            row = image_row(row, folder, templates)
        elif "mm_hint" in row:
            row["mm_hint"]["hint_path"] = locate(folder, row["mm_hint"]["hint_path"])
        else:
            raise ValueError(f"Unsupported row schema: {row.keys()}")
        output.append(row)
    prepared = ROOT / "data/prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    full = prepared / f"{kind}_train.json"
    full.write_text(json.dumps(output, ensure_ascii=False) + "\n")
    # Training smoke samples; explicitly NOT a held-out benchmark.
    groups = defaultdict(list)
    for row in output:
        groups[row["data_source"]].append(row)
    rng = random.Random(42)
    for group in groups.values():
        rng.shuffle(group)
    subset = []
    for i in range(32):
        group = groups[sorted(groups)[i % len(groups)]]
        subset.append(group[(i // len(groups)) % len(group)])
    (prepared / f"{kind}_smoke.json").write_text(json.dumps(subset, ensure_ascii=False) + "\n")
    report = {"kind": kind, "samples": len(output), "source": str(original),
              "prepared": str(full), "all_media_paths_exist": True,
              "data_sources": dict(Counter(row["data_source"] for row in output)),
              "smoke_data_is_training_subset": True}
    (ROOT / "manifests" / f"{kind}-data-validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["image", "video"])
    prepare(parser.parse_args().kind)
