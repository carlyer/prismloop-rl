#!/usr/bin/env python3
"""Exercise upstream imports and an actual persistent Python/image tool call."""
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    from run_training import assert_idle
    assert_idle(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    report = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "versions": {}, "imports": [], "training_completed": False}
    for name in ["torch", "transformers", "vllm", "ray", "flash-attn", "datasets", "tensordict"]:
        report["versions"][name] = importlib.metadata.version(name)
    for name in ["verl.trainer.main_ppo", "verl.workers.fsdp_workers",
                 "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
                 "verl.models.transformers.qwen2_vl",
                 "verl.workers.agent.parallel_env"]:
        importlib.import_module(name)
        report["imports"].append(name)
        print("IMPORT OK", name, flush=True)
    import torch
    from PIL import Image
    from verl.workers.agent.envs.agents_x.safe_persis_python_exe_tool_w_image_hint import MultiModalPythonTool_w_Image_Hint
    tool = MultiModalPythonTool_w_Image_Hint()
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    try:
        tool.reset(raw_prompt="", multi_modal_data={"image": [image]},
                   origin_multi_modal_data={"image": [image]},
                   tool_using_cumulative_reward_per_turn=0.1)
        obs, reward, done, info = tool.execute(
            "<code>```python\nprint(image_clue_0.size)\nplt.imshow(image_clue_0)\nplt.show()\n```</code>")
        if not isinstance(obs, dict) or len(obs["multi_modal_data"]["image"]) != 1 or reward != 0.1 or done:
            raise RuntimeError(f"Tool execution failed: {info}")
        # Verify state persists across calls, then final-answer termination.
        obs2, _, _, _ = tool.execute("<code>```python\npersistent_value = 17\n```</code>")
        obs3, _, _, _ = tool.execute("<code>```python\nprint(persistent_value + 1)\n```</code>")
        if "18" not in str(obs3):
            raise RuntimeError("Persistent Python state did not survive across turns")
        _, _, final_done, final_info = tool.execute("<answer>red</answer>")
        if not final_done or final_info.get("final_answer") != "red":
            raise RuntimeError("Final answer termination failed")
        report["image_tool_smoke"] = {"rendered_images": 1, "persistent_state": True,
                                      "final_answer_termination": True}
    finally:
        tool.close()
    # The caller restricts this to one currently unused GPU.
    if torch.cuda.is_available():
        tensor = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
        value = (tensor @ tensor).float().mean().item()
        assert value == 32.0
        report["cuda"] = {"visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "name": torch.cuda.get_device_name(0), "bf16_matmul": "passed"}
    else:
        raise RuntimeError("CUDA unavailable in reproduction environment")
    report["status"] = "passed"
    (ROOT / "manifests/preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
