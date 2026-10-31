"""Portable inference settings for MIKU."""

from dataclasses import dataclass, field, fields
from pathlib import Path
import os


@dataclass
class MikuConfig:
    device: str = "cuda"
    multiview_backend: str = "miku"
    checkpoints_dir: str = field(default_factory=lambda: os.environ.get("MIKU_CHECKPOINT_DIR", "checkpoints"))
    miku_pretrained_path: str = "flux2-klein-base-9b"
    miku_lora_path: str = "albedo/pytorch_lora_weights.safetensors"
    miku_lora_scale: float = 1.0
    miku_lora_weight_name: str = "pytorch_lora_weights.safetensors"
    miku_steps: int = 10
    miku_guidance_scale: float = 2.0
    miku_seed: int = 0
    miku_dtype: str = "float16"
    miku_offload: str = "model"
    miku_local_files_only: bool = True
    miku_max_canvas_size: int = 2048
    miku_debug_dir: str | None = None
    miku_reference_resize_mode: str = "fit"
    miku_prompt: str = ("Masterpiece, best quality, ultra-detailed, extremely high clarity, sharp focus, "
                        "clean and precise structures, high-frequency details preserved, no blur, no grain, "
                        "no artifacts, high SNR, accurate color reproduction, professional rendering quality")
    pbr_enabled: bool = True
    pbr_checkpoint_path: str = "pbr"
    pbr_pretrained_path: str | None = None  # Use the same base as Miku unless explicitly overridden.
    pbr_dino_weights_path: str = "dinov2/dinov2_vitl14_reg4_pretrain.pth"
    pbr_steps: int = 20
    pbr_seed: int = 99
    pbr_dtype: str = "bfloat16"
    pbr_offload: str = "model"
    pbr_reference_size: int = 512
    pbr_mode: str = "pack"
    pbr_background_white: bool = True
    pbr_vae_tiling: bool = False
    pbr_prompt: str = "turn image to PBR Materials"
    raster_mode: str = "cr"
    bake_mode: str = "back_sample"
    render_size: int = 1024
    resolution: int = 1024
    texture_size: int = 2048
    bake_exp: int = 4
    merge_method: str = "fast"
    reuse_uv: bool = False
    vertex_inpaint: bool = True
    coordinate_system: str = "miku"
    camera_type: str = "perspective"
    camera_distance: float = 4.5
    shader_type: str = "vertex"
    view_selection: str = "fixed"
    max_selected_view_num: int = 6
    # Preserve this camera order through packing and baking.
    candidate_camera_azims: list = field(default_factory=lambda: [270, 90, 0, 180, 0, 0])
    candidate_camera_elevs: list = field(default_factory=lambda: [0, 0, 0, 0, 90, -90])
    candidate_view_weights: list = field(default_factory=lambda: [1.0] * 6)

    def __post_init__(self):
        if self.multiview_backend != "miku" or self.coordinate_system != "miku":
            raise ValueError("MikuConfig requires the miku backend and miku coordinates")
        for name in ("render_size", "resolution", "texture_size", "miku_steps", "miku_max_canvas_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.resolution % 16 or self.miku_max_canvas_size % 16:
            raise ValueError("resolution and miku_max_canvas_size must be multiples of 16")
        if self.miku_offload not in ("none", "model", "sequential"):
            raise ValueError("miku_offload must be none, model or sequential")
        if self.pbr_enabled:
            if self.pbr_steps < 1:
                raise ValueError("pbr_steps must be positive")
            for name in ("pbr_reference_size",):
                if getattr(self, name) < 16 or getattr(self, name) % 16:
                    raise ValueError(f"{name} must be a positive multiple of 16")
            if self.pbr_dtype not in ("float16", "bfloat16"):
                raise ValueError("pbr_dtype must be float16 or bfloat16")
            if self.pbr_offload not in ("none", "model"):
                raise ValueError("pbr_offload must be none or model")
            if self.pbr_mode not in ("pack", "nonpack"):
                raise ValueError("pbr_mode must be pack or nonpack")
            if not isinstance(self.pbr_background_white, bool):
                raise ValueError("pbr_background_white must be a boolean")

    def resolve_paths(self):
        """Resolve relative weights under checkpoints_dir, independent of cwd."""
        root = Path(self.checkpoints_dir).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parent / root
        self.checkpoints_dir = str(root.resolve())
        for name in ("miku_pretrained_path", "miku_lora_path", "pbr_checkpoint_path",
                     "pbr_pretrained_path", "pbr_dino_weights_path"):
            value = getattr(self, name)
            if value:
                path = Path(value).expanduser()
                setattr(self, name, str(path if path.is_absolute() else root / path))
        return self

    def validate_weights(self):
        """Check all local assets before loading models; never fetch weights implicitly."""
        self.resolve_paths()
        base = Path(self.miku_pretrained_path)
        required = [base / "model_index.json", Path(self.miku_lora_path)]
        directories = [base / name for name in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler")]
        if self.pbr_enabled:
            required += [Path(self.pbr_checkpoint_path) / "pytorch_lora_weights.safetensors",
                         Path(self.pbr_checkpoint_path) / "dino_linear.pt", Path(self.pbr_dino_weights_path)]
            if self.pbr_pretrained_path:
                other = Path(self.pbr_pretrained_path)
                required.append(other / "model_index.json")
                directories += [other / name for name in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler")]
        missing = [str(path) for path in required if not path.is_file()]
        missing += [str(path) for path in directories if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Missing MIKU checkpoint assets:\n  " + "\n  ".join(missing)
                                    + "\nPlace the model files under checkpoints/ or use --weights-dir. See README.md.")

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with Path(path).open() as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, dict):
            raise ValueError("Miku configuration must be a YAML mapping")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown Miku settings: {sorted(unknown)}")
        return cls(**values)
