"""FireRed-Image-Edit end-to-end inference entrypoint."""


import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from utils.fast_pipeline import load_fast_pipeline


def str2bool(value: Any) -> bool:
    """Parse loose boolean CLI inputs."""
    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FireRed-Image-Edit inference script"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="FireRedTeam/FireRed-Image-Edit-1.0",
        help="Path to the model or HuggingFace model ID",
    )
    parser.add_argument(
        "--input_image",
        type=Path,
        nargs="+",
        default=[Path("./examples/cola.png")],
        help="Path(s) to the input image(s). Supports 1-N images. "
             "When more than 3 images are given the agent will "
             "automatically crop and stitch them into 2-3 composites.",
    )
    parser.add_argument(
        "--output_image",
        type=Path,
        default=Path("output_edit.png"),
        help="Path to save the output image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transform the object into a realistic miniature product by carefully holding it between your thumb and forefinger.",
        help="Editing prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=49,
        help="Random seed for generation",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="True CFG scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=40,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--recaption",
        action="store_true",
        default=False,
        help="Enable agent-based recaption: expand the editing prompt to "
             "~512 words/characters via Gemini for richer context. "
             "Requires GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--agent_mode",
        type=str,
        choices=["legacy", "runtime"],
        default="legacy",
        help="Agent execution mode. `legacy` keeps the old preprocess-only agent; "
             "`runtime` enables the new planning/tool/workspace/critic runtime.",
    )
    parser.add_argument(
        "--agent_max_repair_rounds",
        type=int,
        default=2,
        help="Maximum automatic repair rounds when --agent_mode runtime is used.",
    )
    parser.add_argument(
        "--agent_max_plan_iterations",
        type=int,
        default=3,
        help="Maximum planner iterations when --agent_mode runtime is used.",
    )
    parser.add_argument(
        "--disable_vlm_critic",
        action="store_true",
        help="Disable the Gemini-backed final critic in runtime agent mode.",
    )
    parser.add_argument(
        "--disable_input_understanding",
        action="store_true",
        help="Disable the optional image-understanding step in runtime agent mode.",
    )
    parser.add_argument(
        "--optimized",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help="Enable Int8, Cache, and Compile. "
             "Supports both `--optimized` and `--optimized True`.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale passed into the diffusion pipeline.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="Negative prompt passed into the diffusion pipeline.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model / LoRA from local files only.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Optional device map for multi-GPU inference, e.g. `balanced` or `auto`.",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=str,
        default=None,
        help="Per visible GPU memory budget, e.g. `22GiB`. "
             "When set and --device_map is omitted, `balanced` is used automatically.",
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=str,
        default="128GiB",
        help="CPU RAM budget used together with --per_gpu_max_memory.",
    )
    parser.add_argument(
        "--generator_device",
        type=str,
        default="auto",
        help="Torch generator device. Defaults to cuda:0 when CUDA is available.",
    )
    parser.add_argument(
        "--enable_attention_slicing",
        action="store_true",
        help="Enable attention slicing to reduce inference memory usage.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional LoRA directory / repo path.",
    )
    parser.add_argument(
        "--lora_weight_name",
        type=str,
        default=None,
        help="Optional LoRA safetensors file name inside --lora_path.",
    )
    parser.add_argument(
        "--lora_adapter_name",
        type=str,
        default="demo",
        help="Adapter name used when loading LoRA weights.",
    )
    parser.add_argument(
        "--fuse_lora",
        action="store_true",
        help="Fuse LoRA weights into the base model after loading. "
             "Recommended for multi-GPU sharded inference.",
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Optional runtime override for GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--gemini_base_url",
        type=str,
        default=None,
        help="Optional runtime override for a Gemini-compatible base URL / API endpoint.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default=None,
        help="Optional runtime override for GEMINI_MODEL_NAME.",
    )
    parser.add_argument(
        "--save_agent_debug_dir",
        type=Path,
        default=None,
        help="Optional directory to save agent composite images and metadata.",
    )
    return parser.parse_args()


def _build_max_memory(
    per_gpu_max_memory: str | None,
    cpu_max_memory: str | None,
) -> dict[Any, str] | None:
    """Build the max_memory mapping for diffusers multi-GPU loading."""
    if per_gpu_max_memory is None:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError("--per_gpu_max_memory requires CUDA to be available.")

    max_memory: dict[Any, str] = {
        idx: per_gpu_max_memory for idx in range(torch.cuda.device_count())
    }
    if cpu_max_memory:
        max_memory["cpu"] = cpu_max_memory
    return max_memory


def _apply_gemini_runtime_overrides(args: argparse.Namespace) -> None:
    """Apply runtime Gemini overrides before importing agent modules."""
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key
    if args.gemini_base_url:
        os.environ["GEMINI_BASE_URL"] = args.gemini_base_url
    if args.gemini_model:
        os.environ["GEMINI_MODEL_NAME"] = args.gemini_model


def _save_agent_debug_output(
    debug_dir: Path,
    input_paths: list[Path],
    original_prompt: str,
    rewritten_prompt: str,
    agent_result: Any,
) -> None:
    """Persist agent composite images and metadata for debugging."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    composite_paths: list[str] = []
    for idx, image in enumerate(agent_result.images, start=1):
        save_path = debug_dir / f"composite_{idx}.png"
        image.save(save_path)
        composite_paths.append(str(save_path.resolve()))

    metadata = {
        "input_images": [str(p.resolve()) for p in input_paths],
        "original_prompt": original_prompt,
        "rewritten_prompt": rewritten_prompt,
        "group_indices": agent_result.group_indices,
        "rois": agent_result.rois,
        "composite_images": composite_paths,
    }
    metadata_path = debug_dir / "agent_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Agent debug artifacts saved at: {debug_dir.resolve()}")


def _save_runtime_debug_output(
    debug_dir: Path,
    input_paths: list[Path],
    original_prompt: str,
    runtime_result: Any,
) -> None:
    """Persist comprehensive-agent trace artifacts for debugging."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    snapshot = runtime_result.workspace_snapshot
    artifact_summaries = snapshot.get("artifacts", {})
    image_paths: dict[str, str] = {}

    for artifact_id, image in runtime_result.debug_image_artifacts.items():
        label = artifact_summaries.get(artifact_id, {}).get("label", artifact_id)
        safe_label = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in label
        )
        save_path = debug_dir / f"{artifact_id}_{safe_label}.png"
        image.save(save_path)
        image_paths[artifact_id] = str(save_path.resolve())

    metadata = {
        "input_images": [str(path.resolve()) for path in input_paths],
        "original_prompt": original_prompt,
        "final_prompt": runtime_result.final_prompt,
        "final_status": runtime_result.final_status,
        "critic_summary": runtime_result.critic_summary,
        "group_indices": runtime_result.group_indices,
        "rois": runtime_result.rois,
        "execution_trace": runtime_result.execution_trace,
        "plans": runtime_result.plans,
        "workspace_snapshot": snapshot,
        "saved_images": image_paths,
    }
    metadata_path = debug_dir / "runtime_agent_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Runtime agent debug artifacts saved at: {debug_dir.resolve()}")


def load_pipeline(args: argparse.Namespace) -> QwenImageEditPlusPipeline:
    """Load FireRed image edit pipeline with optional LoRA / multi-GPU settings."""
    if args.optimized:
        if any(
            [
                args.lora_path,
                args.device_map,
                args.per_gpu_max_memory,
                args.local_files_only,
            ]
        ):
            raise ValueError(
                "--optimized currently only supports direct single-GPU loading "
                "without LoRA or device_map sharding."
            )
        pipe = load_fast_pipeline(args.model_path)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    device_map = args.device_map
    if device_map is None and args.per_gpu_max_memory:
        device_map = "balanced"
        print("No --device_map specified; using `balanced` because --per_gpu_max_memory was set.")

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
    }
    if args.local_files_only:
        load_kwargs["local_files_only"] = True
    if device_map:
        load_kwargs["device_map"] = device_map

    max_memory = _build_max_memory(args.per_gpu_max_memory, args.cpu_max_memory)
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_path,
        **load_kwargs,
    )

    if not device_map:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(target_device)
        print(f"Pipeline moved to {target_device}.")
    else:
        print(f"Pipeline loaded with device_map={device_map!r}.")

    if args.lora_path:
        lora_kwargs: dict[str, Any] = {
            "adapter_name": args.lora_adapter_name,
        }
        if args.lora_weight_name:
            lora_kwargs["weight_name"] = args.lora_weight_name
        if args.local_files_only:
            lora_kwargs["local_files_only"] = True

        pipe.load_lora_weights(args.lora_path, **lora_kwargs)
        print(f"LoRA loaded from: {args.lora_path}")

        if args.fuse_lora:
            pipe.fuse_lora()
            print("LoRA weights fused into the base model.")

    if args.enable_attention_slicing or device_map:
        pipe.enable_attention_slicing()
        print("Attention slicing enabled.")

    pipe.set_progress_bar_config(disable=None)
    return pipe



def main() -> None:
    """Main entry point."""
    args = parse_args()

    # ── Load all input images ──
    images = [Image.open(p).convert("RGB") for p in args.input_image]
    prompt = args.prompt
    print(f"Loaded {len(images)} image(s).")


    # ── Agent: either legacy preprocess-only mode or runtime closed loop ──
    need_stitch = len(images) > 3
    need_recaption = args.recaption

    if args.agent_mode == "runtime":
        _apply_gemini_runtime_overrides(args)
        from agent import (
            AgentRuntime,
            AgentRuntimeOptions,
            FireRedEditConfig,
            FireRedEditTool,
        )

        pipeline = load_pipeline(args)
        print("Pipeline loaded for runtime agent.")

        runtime = AgentRuntime(
            edit_tool=FireRedEditTool(
                pipeline,
                FireRedEditConfig(
                    num_inference_steps=args.num_inference_steps,
                    true_cfg_scale=args.true_cfg_scale,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=args.negative_prompt,
                    seed=args.seed,
                    generator_device=args.generator_device,
                ),
            ),
            verbose=True,
        )
        runtime_result = runtime.run(
            images,
            prompt,
            options=AgentRuntimeOptions(
                enable_recaption=need_recaption or need_stitch,
                max_repair_rounds=args.agent_max_repair_rounds,
                max_plan_iterations=args.agent_max_plan_iterations,
                enable_input_understanding=not args.disable_input_understanding,
                enable_vlm_critic=not args.disable_vlm_critic,
            ),
        )
        if args.save_agent_debug_dir is not None:
            _save_runtime_debug_output(
                args.save_agent_debug_dir,
                args.input_image,
                args.prompt,
                runtime_result,
            )

        args.output_image.parent.mkdir(parents=True, exist_ok=True)
        runtime_result.final_images[0].save(args.output_image)
        print("Runtime agent status:", runtime_result.final_status)
        print("Image saved at:", args.output_image.resolve())
        return

    if need_stitch or need_recaption:
        _apply_gemini_runtime_overrides(args)
        from agent import AgentPipeline

        agent = AgentPipeline(verbose=True)
        agent_result = agent.run(
            images,
            prompt,
            enable_recaption=need_recaption or need_stitch,
        )
        images = agent_result.images
        prompt = agent_result.prompt
        print(f"Agent produced {len(images)} image(s).")
        print(f"Rewritten prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
        if args.save_agent_debug_dir is not None:
            _save_agent_debug_output(
                args.save_agent_debug_dir,
                args.input_image,
                args.prompt,
                prompt,
                agent_result,
            )

    pipeline = load_pipeline(args)
    print("Pipeline loaded.")

    generator_device = args.generator_device
    if generator_device == "auto":
        generator_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using generator device: {generator_device}")


    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.Generator(device=generator_device).manual_seed(args.seed),
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": 1,
    }


    if args.optimized:
        print("NOTE: The first inference after compilation may take 1-2 minutes.")


    with torch.inference_mode():
        result = pipeline(**inputs)


    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    output_image = result.images[0]
    output_image.save(args.output_image)


    print("Image saved at:", args.output_image.resolve())
    
    # ── Replace with the desired case or scenario based on your specific needs ── 
    if args.optimized:
        print("Subsequent runs will be significantly faster. Enjoy~")
        with torch.inference_mode():
            result = pipeline(**inputs)
        output_image = result.images[0]
        output_image.save(args.output_image)



if __name__ == "__main__":
    main()
