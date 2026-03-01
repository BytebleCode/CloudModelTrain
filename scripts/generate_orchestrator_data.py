#!/usr/bin/env python3
"""
Generate training data for the orchestrator agent.

Combines real HuggingFace datasets (SWE trajectories, function-calling,
task decomposition) with synthetic resource-management scenarios to teach
the orchestrator how to coordinate specialist agents on GPU hardware.

Usage:
    python scripts/generate_orchestrator_data.py
    python scripts/generate_orchestrator_data.py --max_samples 5000
    python scripts/generate_orchestrator_data.py --synthetic_only --num_synthetic 2000

Output: datasets/orchestrator/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gen_orchestrator")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "orchestrator"

# ---------------------------------------------------------------------------
# Agent definitions (real model sizes at Q4 quantization)
# ---------------------------------------------------------------------------

AGENTS = {
    "code_writer": {
        "model": "Devstral-24B",
        "vram_gb": 14,
        "description": "Generates production Python code from specifications",
    },
    "test_generator": {
        "model": "Devstral-24B",
        "vram_gb": 14,
        "description": "Creates comprehensive Python test suites",
    },
    "static_reviewer": {
        "model": "Qwen3-32B",
        "vram_gb": 18,
        "description": "Reviews Python code for quality, bugs, and style",
    },
    "security_auditor": {
        "model": "Qwen3-32B",
        "vram_gb": 18,
        "description": "Identifies security vulnerabilities in Python code",
    },
    "performance_optimizer": {
        "model": "Qwen3-32B",
        "vram_gb": 18,
        "description": "Optimizes Python code for speed and memory efficiency",
    },
    "docs_generator": {
        "model": "Qwen2.5-7B",
        "vram_gb": 4,
        "description": "Generates Python documentation and docstrings",
    },
}

GPU_PROFILES = {
    "A100 PCIe 80GB": {"vram_gb": 80, "cost_per_hr": 1.14},
    "H200 SXM 141GB": {"vram_gb": 141, "cost_per_hr": 3.29},
}

# Overhead: CUDA context, KV cache, activations — reserve ~6GB
GPU_OVERHEAD_GB = 6


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    gpu_name: str,
    loaded_agents: dict[str, int],
    running_tasks: list[str],
) -> str:
    """Build the orchestrator system prompt with current resource state."""
    gpu = GPU_PROFILES[gpu_name]
    total_vram = gpu["vram_gb"]
    used_vram = sum(loaded_agents.values()) + GPU_OVERHEAD_GB
    available_vram = total_vram - used_vram

    agent_list = "\n".join(
        f"- {name} ({info['model']}, ~{info['vram_gb']}GB VRAM Q4): {info['description']}"
        for name, info in AGENTS.items()
    )

    loaded_str = ", ".join(
        f"{name} ({vram}GB)" for name, vram in loaded_agents.items()
    ) if loaded_agents else "none"

    running_str = "\n".join(f"- {t}" for t in running_tasks) if running_tasks else "- none"

    return (
        "You are a resource-aware orchestrator managing coding agents on GPU hardware.\n"
        "\n"
        "Available agents:\n"
        f"{agent_list}\n"
        "\n"
        "Current GPU state:\n"
        f"- GPU: {gpu_name} ({total_vram}GB VRAM)\n"
        f"- Loaded: {loaded_str}\n"
        f"- Available VRAM: {available_vram}GB\n"
        f"- Cost: ${gpu['cost_per_hr']}/hr\n"
        f"- Running tasks:\n{running_str}\n"
        "\n"
        "Respond with a JSON execution plan. Each step has: step number, agent, "
        "action (load/run/unload), reason, and optional depends_on. "
        "Include estimated_vram_peak and cost_estimate."
    )


# ---------------------------------------------------------------------------
# Synthetic resource management scenarios
# ---------------------------------------------------------------------------

# Task templates that map to agent pipelines
TASK_TEMPLATES = [
    {
        "task": "Write a Python function that {func_desc}, then generate unit tests for it.",
        "pipeline": ["code_writer", "test_generator"],
        "func_descs": [
            "parses CSV files with custom delimiters",
            "implements a binary search tree",
            "handles HTTP retry logic with exponential backoff",
            "validates email addresses using regex",
            "implements an LRU cache decorator",
            "reads and writes Parquet files",
            "creates a thread-safe connection pool",
            "implements the observer pattern",
            "serializes Python objects to MessagePack",
            "builds a simple CLI argument parser",
        ],
    },
    {
        "task": "Review the {module} module for security vulnerabilities, then generate updated documentation.",
        "pipeline": ["security_auditor", "docs_generator"],
        "modules": [
            "authentication", "payment processing", "user session management",
            "file upload handler", "API gateway", "database migration",
            "token validation", "password hashing", "OAuth callback",
            "webhook receiver",
        ],
    },
    {
        "task": "Review the {module} module for code quality, optimize performance, and update docs.",
        "pipeline": ["static_reviewer", "performance_optimizer", "docs_generator"],
        "modules": [
            "data pipeline", "image processing", "search indexer",
            "recommendation engine", "batch job scheduler", "log aggregator",
            "metrics collector", "cache invalidation", "rate limiter",
            "message queue consumer",
        ],
    },
    {
        "task": "Audit the {module} for security, review code quality, then write tests.",
        "pipeline": ["security_auditor", "static_reviewer", "test_generator"],
        "modules": [
            "REST API endpoints", "GraphQL resolvers", "middleware stack",
            "input sanitizer", "CORS handler", "JWT token service",
            "role-based access control", "API key manager",
        ],
    },
    {
        "task": "Write a {component}, optimize it for performance, audit for security, and document it.",
        "pipeline": ["code_writer", "performance_optimizer", "security_auditor", "docs_generator"],
        "components": [
            "database connection pool manager",
            "distributed task queue",
            "real-time WebSocket handler",
            "file encryption service",
            "rate-limiting middleware",
        ],
    },
    {
        "task": "Generate comprehensive tests for the {module}, then review the tests for quality.",
        "pipeline": ["test_generator", "static_reviewer"],
        "modules": [
            "payment service", "user registration flow", "notification system",
            "inventory tracker", "report generator", "email sender",
        ],
    },
]


def _pick_template_task(template: dict) -> str:
    """Fill in a task template with random values."""
    task = template["task"]
    if "{func_desc}" in task:
        task = task.format(func_desc=random.choice(template["func_descs"]))
    elif "{module}" in task:
        task = task.format(module=random.choice(template["modules"]))
    elif "{component}" in task:
        task = task.format(component=random.choice(template["components"]))
    return task


def _build_plan(
    pipeline: list[str],
    loaded_agents: dict[str, int],
    gpu_name: str,
    task_description: str,
) -> dict:
    """Build a realistic orchestrator response plan."""
    gpu = GPU_PROFILES[gpu_name]
    total_vram = gpu["vram_gb"]
    used = sum(loaded_agents.values()) + GPU_OVERHEAD_GB
    steps = []
    step_num = 0
    agents_to_unload_after = []
    peak_vram = used

    for i, agent_name in enumerate(pipeline):
        agent = AGENTS[agent_name]
        needed_vram = agent["vram_gb"]

        # Load if not already loaded
        if agent_name not in loaded_agents:
            available = total_vram - used
            # Check if we need to unload something
            if needed_vram > available:
                # Find least-needed loaded agent not in our pipeline
                for loaded_name, loaded_vram in list(loaded_agents.items()):
                    if loaded_name not in pipeline:
                        step_num += 1
                        steps.append({
                            "step": step_num,
                            "agent": loaded_name,
                            "action": "unload",
                            "reason": f"Need {needed_vram}GB for {agent_name}, freeing {loaded_vram}GB",
                        })
                        used -= loaded_vram
                        del loaded_agents[loaded_name]
                        if used + needed_vram <= total_vram:
                            break

            step_num += 1
            steps.append({
                "step": step_num,
                "agent": agent_name,
                "action": "load",
                "reason": f"Need {agent_name} ({needed_vram}GB). Available VRAM: {total_vram - used}GB. Safe to load.",
            })
            used += needed_vram
            loaded_agents[agent_name] = needed_vram
            agents_to_unload_after.append(agent_name)

        peak_vram = max(peak_vram, used)

        # Run step
        step_num += 1
        run_step: dict[str, Any] = {
            "step": step_num,
            "agent": agent_name,
            "action": "run",
            "input": _make_agent_input(agent_name, task_description),
        }
        # Add dependency on previous agent's run step
        if i > 0:
            # depends on the previous run step
            for prev_step in reversed(steps):
                if prev_step["action"] == "run":
                    run_step["depends_on"] = prev_step["step"]
                    break
        steps.append(run_step)

    # Unload agents that were loaded just for this task
    for agent_name in agents_to_unload_after:
        if agent_name in loaded_agents:
            step_num += 1
            steps.append({
                "step": step_num,
                "agent": agent_name,
                "action": "unload",
                "reason": f"Task complete, free {loaded_agents[agent_name]}GB VRAM",
            })

    # Cost estimate: ~30s per agent step at gpu rate
    est_minutes = len([s for s in steps if s["action"] == "run"]) * 0.5
    cost_est = round(est_minutes / 60 * gpu["cost_per_hr"], 2)

    return {
        "plan": steps,
        "estimated_vram_peak": f"{peak_vram}GB",
        "cost_estimate": f"${cost_est}",
    }


def _make_agent_input(agent_name: str, task_description: str) -> str:
    """Generate a plausible input description for an agent step."""
    prefixes = {
        "code_writer": "Generate production Python code: ",
        "test_generator": "Write comprehensive unit tests for: ",
        "static_reviewer": "Review code quality for: ",
        "security_auditor": "Audit for security vulnerabilities: ",
        "performance_optimizer": "Optimize performance of: ",
        "docs_generator": "Generate documentation for: ",
    }
    return prefixes.get(agent_name, "Process: ") + task_description


def generate_synthetic_scenarios(num_scenarios: int = 2000) -> list[dict]:
    """Generate synthetic resource management training examples."""
    records = []

    for _ in range(num_scenarios):
        # Random GPU
        gpu_name = random.choice(list(GPU_PROFILES.keys()))
        gpu = GPU_PROFILES[gpu_name]

        # Random set of currently loaded agents (0-3)
        num_loaded = random.randint(0, 3)
        all_agent_names = list(AGENTS.keys())
        loaded_names = random.sample(all_agent_names, min(num_loaded, len(all_agent_names)))
        loaded_agents = {name: AGENTS[name]["vram_gb"] for name in loaded_names}

        # Random running tasks (0-1)
        running_tasks = []
        if loaded_agents and random.random() > 0.4:
            running_agent = random.choice(list(loaded_agents.keys()))
            task_num = random.randint(1, 20)
            running_tasks.append(f"{running_agent} processing task #{task_num}")

        # Pick a task template
        template = random.choice(TASK_TEMPLATES)
        task_text = _pick_template_task(template)
        pipeline = template["pipeline"]

        # Build system prompt
        system_prompt = build_system_prompt(gpu_name, dict(loaded_agents), running_tasks)

        # Build response plan
        plan = _build_plan(pipeline, dict(loaded_agents), gpu_name, task_text)

        records.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"New task: {task_text}"},
                {"role": "assistant", "content": json.dumps(plan, indent=None)},
            ]
        })

    return records


# ---------------------------------------------------------------------------
# HuggingFace dataset downloaders + reformatters
# ---------------------------------------------------------------------------

def download_swe_trajectories(max_samples: int | None = None) -> list[dict]:
    """
    Download SWE-agent trajectories and reformat as orchestrator dispatch plans.
    Maps think-act-observe loops to agent coordination patterns.
    """
    from datasets import load_dataset

    records = []

    # --- nebius/SWE-agent-trajectories ---
    logger.info("Loading nebius/SWE-agent-trajectories...")
    try:
        ds = load_dataset("nebius/SWE-agent-trajectories", split="train")
        count = 0
        for row in ds:
            if max_samples and count >= max_samples // 2:
                break

            # Extract trajectory content
            trajectory = row.get("trajectory") or row.get("text") or ""
            if isinstance(trajectory, list):
                trajectory = "\n".join(str(t) for t in trajectory[:10])
            elif not isinstance(trajectory, str):
                continue
            if len(trajectory) < 100:
                continue

            issue = row.get("issue") or row.get("problem_statement") or ""
            if isinstance(issue, dict):
                issue = issue.get("title", "") + "\n" + issue.get("body", "")
            if not issue or len(str(issue).strip()) < 20:
                continue

            # Map to orchestrator format: SWE workflow -> agent dispatch
            pipeline = _infer_pipeline_from_text(str(issue))
            gpu_name = random.choice(list(GPU_PROFILES.keys()))

            system_prompt = build_system_prompt(gpu_name, {}, [])
            plan = _build_plan(pipeline, {}, gpu_name, str(issue)[:200])

            records.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Resolve this issue:\n{str(issue)[:500]}"},
                    {"role": "assistant", "content": json.dumps(plan, indent=None)},
                ]
            })
            count += 1

    except Exception as e:
        logger.warning("SWE-agent-trajectories failed: %s", e)

    # --- SWE-bench/SWE-smith-trajectories ---
    logger.info("Loading SWE-bench/SWE-smith-trajectories...")
    try:
        ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="train")
        count = 0
        for row in ds:
            if max_samples and count >= max_samples // 2:
                break

            problem = row.get("problem_statement") or row.get("text") or ""
            if not problem or len(str(problem).strip()) < 20:
                continue

            pipeline = _infer_pipeline_from_text(str(problem))
            gpu_name = random.choice(list(GPU_PROFILES.keys()))

            system_prompt = build_system_prompt(gpu_name, {}, [])
            plan = _build_plan(pipeline, {}, gpu_name, str(problem)[:200])

            records.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Resolve this issue:\n{str(problem)[:500]}"},
                    {"role": "assistant", "content": json.dumps(plan, indent=None)},
                ]
            })
            count += 1

    except Exception as e:
        logger.warning("SWE-smith-trajectories failed: %s", e)

    logger.info("SWE trajectories: %d records", len(records))
    return records


def download_function_calling(max_samples: int | None = None) -> list[dict]:
    """
    Download function-calling datasets and reformat with agent tool definitions.
    """
    from datasets import load_dataset

    records = []

    # Agent tools definition for function-calling format
    agent_tools = [
        {
            "name": name,
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Task description"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["input"],
            },
        }
        for name, info in AGENTS.items()
    ]
    tools_json = json.dumps(agent_tools, indent=None)

    # --- NousResearch/hermes-function-calling-v1 ---
    logger.info("Loading NousResearch/hermes-function-calling-v1...")
    try:
        ds = load_dataset("NousResearch/hermes-function-calling-v1", split="train")
        count = 0
        for row in ds:
            if max_samples and count >= max_samples // 2:
                break

            conversations = row.get("conversations") or row.get("messages") or []
            if not conversations or len(conversations) < 2:
                continue

            # Reformat: replace original tools with our agent tools
            messages = []
            messages.append({
                "role": "system",
                "content": (
                    "You are a coding agent orchestrator. You coordinate specialist agents "
                    "by calling them as tools.\n\n"
                    f"Available tools:\n{tools_json}"
                ),
            })
            for msg in conversations:
                role = msg.get("role") or msg.get("from", "user")
                content = msg.get("content") or msg.get("value") or ""
                if role in ("system", "tool"):
                    continue
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                else:
                    continue
                if content:
                    messages.append({"role": role, "content": str(content)[:1000]})

            if len(messages) >= 3:
                records.append({"messages": messages})
                count += 1

    except Exception as e:
        logger.warning("hermes-function-calling-v1 failed: %s", e)

    # --- glaiveai/glaive-function-calling-v2 ---
    logger.info("Loading glaiveai/glaive-function-calling-v2...")
    try:
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
        count = 0
        for row in ds:
            if max_samples and count >= max_samples // 2:
                break

            # This dataset uses "system"/"chat" text format
            system_text = row.get("system") or ""
            chat_text = row.get("chat") or ""
            if not chat_text:
                continue

            messages = [{
                "role": "system",
                "content": (
                    "You are a coding agent orchestrator with access to specialist agents. "
                    "Coordinate tasks by selecting and invoking the right agent.\n\n"
                    f"Available agents:\n{tools_json}"
                ),
            }]

            # Parse chat turns from text format
            parts = chat_text.split("USER: ")
            for part in parts[1:]:  # skip empty first
                if "ASSISTANT: " in part:
                    user_part, *assistant_parts = part.split("ASSISTANT: ")
                    if user_part.strip():
                        messages.append({"role": "user", "content": user_part.strip()[:1000]})
                    for ap in assistant_parts:
                        cleaned = ap.strip()
                        # Strip function call markers but keep content
                        cleaned = cleaned.replace("<functioncall>", "").replace("</functioncall>", "")
                        if cleaned:
                            messages.append({"role": "assistant", "content": cleaned[:1000]})

            if len(messages) >= 3:
                records.append({"messages": messages})
                count += 1

    except Exception as e:
        logger.warning("glaive-function-calling-v2 failed: %s", e)

    logger.info("Function-calling: %d records", len(records))
    return records


def download_task_decomposition(max_samples: int | None = None) -> list[dict]:
    """
    Download task decomposition dataset and reformat for orchestrator planning.
    """
    from datasets import load_dataset

    records = []

    logger.info("Loading PersonalAILab/TaskCraft...")
    try:
        ds = load_dataset("PersonalAILab/TaskCraft", split="train")
        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break

            task = row.get("task") or row.get("instruction") or row.get("input") or ""
            decomposition = row.get("decomposition") or row.get("output") or row.get("response") or ""

            if not task or not decomposition:
                continue
            if len(str(task).strip()) < 20:
                continue

            # Reframe as orchestrator task decomposition
            gpu_name = random.choice(list(GPU_PROFILES.keys()))
            system_prompt = build_system_prompt(gpu_name, {}, [])

            # Map task decomposition to agent dispatch
            pipeline = _infer_pipeline_from_text(str(task))
            plan = _build_plan(pipeline, {}, gpu_name, str(task)[:200])

            records.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Plan and execute: {str(task)[:500]}"},
                    {"role": "assistant", "content": json.dumps(plan, indent=None)},
                ]
            })
            count += 1

    except Exception as e:
        logger.warning("TaskCraft failed: %s", e)

    logger.info("Task decomposition: %d records", len(records))
    return records


def _infer_pipeline_from_text(text: str) -> list[str]:
    """Heuristically map issue/task text to a pipeline of specialist agents."""
    text_lower = text.lower()
    pipeline = []

    # Check for security keywords
    has_security = any(kw in text_lower for kw in [
        "security", "vulnerab", "auth", "injection", "xss", "csrf",
        "password", "encrypt", "token", "permission", "cve", "cwe",
    ])
    # Check for performance keywords
    has_perf = any(kw in text_lower for kw in [
        "performance", "optimiz", "slow", "fast", "speed", "memory",
        "efficient", "bottleneck", "profil", "complexity",
    ])
    # Check for test keywords
    has_test = any(kw in text_lower for kw in [
        "test", "assert", "unittest", "pytest", "coverage", "mock",
    ])
    # Check for docs keywords
    has_docs = any(kw in text_lower for kw in [
        "document", "docstring", "readme", "api doc", "comment",
    ])
    # Check for code generation keywords
    has_code = any(kw in text_lower for kw in [
        "implement", "write", "create", "add", "build", "function",
        "class", "module", "feature", "fix", "bug", "error",
    ])
    # Check for review keywords
    has_review = any(kw in text_lower for kw in [
        "review", "quality", "lint", "style", "refactor", "clean",
    ])

    # Build pipeline based on detected intent
    if has_code:
        pipeline.append("code_writer")
    if has_review:
        pipeline.append("static_reviewer")
    if has_security:
        pipeline.append("security_auditor")
    if has_perf:
        pipeline.append("performance_optimizer")
    if has_test:
        pipeline.append("test_generator")
    if has_docs:
        pipeline.append("docs_generator")

    # Default: code_writer + test_generator if nothing matched
    if not pipeline:
        pipeline = ["code_writer", "test_generator"]

    return pipeline


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_orchestrator_dataset(
    max_samples: int | None = None,
    num_synthetic: int = 2000,
    synthetic_only: bool = False,
) -> int:
    """Build the full orchestrator training dataset."""
    all_records: list[dict] = []

    # --- Synthetic resource management scenarios (always generated) ---
    logger.info("Generating %d synthetic resource management scenarios...", num_synthetic)
    synthetic = generate_synthetic_scenarios(num_synthetic)
    all_records.extend(synthetic)
    logger.info("Synthetic scenarios: %d records", len(synthetic))

    if not synthetic_only:
        # --- Real HF datasets ---
        per_source_limit = max_samples // 4 if max_samples else None

        try:
            swe = download_swe_trajectories(per_source_limit)
            all_records.extend(swe)
        except ImportError:
            logger.warning("'datasets' library not installed — skipping SWE trajectories")
        except Exception as e:
            logger.warning("SWE trajectories download failed: %s", e)

        try:
            fc = download_function_calling(per_source_limit)
            all_records.extend(fc)
        except ImportError:
            logger.warning("'datasets' library not installed — skipping function-calling data")
        except Exception as e:
            logger.warning("Function-calling download failed: %s", e)

        try:
            td = download_task_decomposition(per_source_limit)
            all_records.extend(td)
        except ImportError:
            logger.warning("'datasets' library not installed — skipping task decomposition data")
        except Exception as e:
            logger.warning("Task decomposition download failed: %s", e)

    # Shuffle and limit
    random.shuffle(all_records)
    if max_samples:
        all_records = all_records[:max_samples]

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Orchestrator dataset: %d records -> %s", len(all_records), output_path)
    return len(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="Generate orchestrator training data"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit total dataset size",
    )
    parser.add_argument(
        "--num_synthetic", type=int, default=2000,
        help="Number of synthetic resource management scenarios (default: 2000)",
    )
    parser.add_argument(
        "--synthetic_only", action="store_true",
        help="Only generate synthetic data (skip HF downloads)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    count = build_orchestrator_dataset(
        max_samples=args.max_samples,
        num_synthetic=args.num_synthetic,
        synthetic_only=args.synthetic_only,
    )
    logger.info("Done. Total: %d records", count)


if __name__ == "__main__":
    main()
