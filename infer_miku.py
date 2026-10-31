"""Render, predict multiview albedo then MR, and bake a PBR textured mesh."""

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "cfgs/miku.yaml"))
    parser.add_argument("--weights-dir", help="Checkpoint root; default: checkpoints/ or MIKU_CHECKPOINT_DIR")
    parser.add_argument("--mesh", help="Input mesh; default: bundled assets/axe/mesh.glb")
    parser.add_argument("--image", help="Reference image; default: bundled assets/axe/image.png; required with a custom --mesh unless --conditions-only")
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs/axe/textured_mesh.obj"),
                        help="Output .obj file; default: outputs/axe/textured_mesh.obj; debug images go beside it")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--conditions-only", action="store_true", help="Render normal/position/masks without loading FLUX")
    mode.add_argument("--mv-only", action="store_true", help="Save predicted views without UV baking")
    parser.add_argument("--debug-dir", help="Default: <output stem>_debug")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--lora-scale", type=float)
    parser.add_argument("--checkpoint")
    parser.add_argument("--lora")
    parser.add_argument("--offload", choices=("none", "model", "sequential"))
    parser.add_argument("--device")
    parser.add_argument("--resolution", type=int, help="Per-view working resolution (multiple of 16)")
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--texture-size", type=int)
    parser.add_argument("--prompt")
    parser.add_argument("--albedo-only", action="store_true", help="Skip PBR inference and export color only")
    parser.add_argument("--pbr-checkpoint", help="Directory containing PBR LoRA and dino_linear.pt")
    parser.add_argument("--pbr-steps", type=int)
    parser.add_argument("--pbr-seed", type=int)
    parser.add_argument("--pbr-mode", choices=("pack", "nonpack"),
                        help="Default: pack (albedo atlas resized by 0.5); nonpack: 1536x1024 grid")
    parser.add_argument("--pbr-background-white", action=argparse.BooleanOptionalAction, default=None,
                        help="Whiten MR background using geometry masks (default: enabled)")
    parser.add_argument("--no-glb", action="store_true")
    args = parser.parse_args(argv)
    if args.mesh is None:
        if args.image is not None:
            parser.error("--mesh is required when --image is supplied")
        args.mesh = str(REPO_ROOT / "assets/axe/mesh.glb")
        if not args.conditions_only:
            args.image = str(REPO_ROOT / "assets/axe/image.png")
    elif not args.conditions_only and not args.image:
        parser.error("--image is required with a custom --mesh for prediction")
    if not Path(args.mesh).is_file():
        parser.error(f"Mesh does not exist: {args.mesh}")
    if args.image and not Path(args.image).is_file():
        parser.error(f"Reference image does not exist: {args.image}")
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".obj":
        parser.error("--output must be an .obj path; .glb is exported alongside it")
    return args


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    from miku_config import MikuConfig
    config = MikuConfig.from_yaml(args.config)
    for source, target in {
        "steps": "miku_steps", "seed": "miku_seed", "guidance": "miku_guidance_scale",
        "lora_scale": "miku_lora_scale", "checkpoint": "miku_pretrained_path",
        "lora": "miku_lora_path", "offload": "miku_offload", "device": "device",
        "resolution": "resolution", "render_size": "render_size", "texture_size": "texture_size",
        "pbr_checkpoint": "pbr_checkpoint_path", "pbr_steps": "pbr_steps", "pbr_seed": "pbr_seed",
        "pbr_mode": "pbr_mode",
        "pbr_background_white": "pbr_background_white",
        "weights_dir": "checkpoints_dir",
    }.items():
        if getattr(args, source) is not None:
            setattr(config, target, getattr(args, source))
    if args.albedo_only:
        config.pbr_enabled = False
    config.miku_debug_dir = str(Path(args.debug_dir).resolve() if args.debug_dir else output.with_suffix("")) + ("" if args.debug_dir else "_debug")
    config.__post_init__()
    if not args.conditions_only:
        config.validate_weights()
    output.parent.mkdir(parents=True, exist_ok=True)
    from mikuPipeline import MikuPipeline
    # Render first so camera/configuration errors don't require loading 9B weights.
    pipeline = MikuPipeline(config, load_models=False)
    conditions = pipeline.prepare_conditions(args.mesh, use_remesh=False)
    if args.conditions_only:
        print(f"Conditions saved to {config.miku_debug_dir}")
        return
    pipeline.load_models()
    views = pipeline.predict_multiview(args.image, conditions, prompt=args.prompt)
    if args.mv_only:
        for channel, images in views.items():
            for index, image in enumerate(images):
                image.save(str(Path(config.miku_debug_dir) / f"{channel}_{index:02d}.png"))
        print(f"Predictions saved to {config.miku_debug_dir}")
    else:
        result = pipeline.bake_multiview(views, conditions, str(output), save_glb=not args.no_glb)
        print(f"Textured mesh saved to {result}")


if __name__ == "__main__":
    main()
