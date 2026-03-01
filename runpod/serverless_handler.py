"""
RunPod Serverless handler — for running training jobs via RunPod's serverless API.

This lets you trigger training jobs programmatically without SSH.

Deploy as a serverless endpoint on RunPod, then call:
    curl -X POST https://api.runpod.ai/v2/<endpoint_id>/run \
      -H "Authorization: Bearer <api_key>" \
      -H "Content-Type: application/json" \
      -d '{"input": {"agent": "code_writer", "run_name": "v1"}}'
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("runpod_handler")


def handler(event: dict) -> dict:
    """RunPod serverless handler entry point."""

    job_input = event.get("input", {})
    agent = job_input.get("agent")
    run_name = job_input.get("run_name", None)
    extra_args = job_input.get("extra_args", [])

    if not agent:
        return {"error": "Missing required field: 'agent'"}

    # Build command
    cmd = [sys.executable, "train.py", "--agent", agent]
    if run_name:
        cmd.extend(["--run_name", run_name])
    if job_input.get("flash_attn", False):
        cmd.append("--flash_attn")
    if job_input.get("resume"):
        cmd.extend(["--resume", job_input["resume"]])
    if job_input.get("lr"):
        cmd.extend(["--lr", str(job_input["lr"])])
    if job_input.get("epochs"):
        cmd.extend(["--epochs", str(job_input["epochs"])])
    cmd.extend(extra_args)

    logger.info("Running: %s", " ".join(cmd))

    # Run training
    project_dir = "/workspace/CloudModelTrain"
    result = subprocess.run(
        cmd,
        cwd=project_dir,
        capture_output=True,
        text=True,
    )

    output = {
        "agent": agent,
        "run_name": run_name,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-5000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
    }

    if result.returncode == 0:
        output["status"] = "success"
        # Find output dir
        for line in result.stdout.split("\n"):
            if "Adapter:" in line:
                output["adapter_path"] = line.split("Adapter:")[-1].strip()
                break
    else:
        output["status"] = "failed"

    return output


# RunPod serverless expects this at module level
if __name__ == "__main__":
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        # Local testing
        test_event = {"input": {"agent": "code_writer", "run_name": "test"}}
        print(handler(test_event))
