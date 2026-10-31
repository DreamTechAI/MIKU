import os
import torch
import json
import trimesh
import numpy as np
from PIL import Image
from DifferentiableRenderer.MeshRender import MeshRender
from utils.pipeline_utils import ViewProcessor
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging
from miku_config import MikuConfig

diffusers_logging.set_verbosity(50)




class MaterialMVPPipeline:

    def __init__(self, config=None, load_models=True) -> None:
        self.config = config if config is not None else MikuConfig()
        self.models = {}
        self.stats_logs = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
            device=self.config.device,
            coordinate_system=getattr(self.config, "coordinate_system", "materialmvp"),
            camera_type=getattr(self.config, "camera_type", "orth"),
            camera_distance=getattr(self.config, "camera_distance", 1.45),
            shader_type=getattr(self.config, "shader_type", "face"),
        )
        self.view_processor = ViewProcessor(self.config, self.render)
        if load_models:
            self.load_models()

    def load_models(self):
        torch.cuda.empty_cache()
        from utils.miku_multiview_utils import MikuMultiviewDiffusionNet
        self.models["multiview_model"] = MikuMultiviewDiffusionNet(self.config)
        print("Models Loaded.")

    @torch.no_grad()
    def prepare_conditions(self, mesh_path, use_remesh=False):
        """Render conditioning and keep the very same cameras for UV projection."""
        mesh_path = os.fspath(mesh_path)
        processed_mesh_path = mesh_path
        if use_remesh:
            raise ValueError("Remeshing is not part of this inference release; supply the final mesh.")
        mesh = trimesh.load(processed_mesh_path, force="mesh")
        uv = getattr(mesh.visual, "uv", None)
        valid_uv = uv is not None and np.shape(uv) == (len(mesh.vertices), 2) and np.isfinite(uv).all()
        if not (getattr(self.config, "reuse_uv", False) and valid_uv):
            from utils.uvwrap_utils import mesh_uv_wrap
            mesh = mesh_uv_wrap(mesh)
        # A pipeline instance can be reused for more than one mesh.
        for attr in ("tex", "tex_mr", "tex_normalMap"):
            if hasattr(self.render, attr):
                delattr(self.render, attr)
        self.render.load_mesh(mesh=mesh)
        if getattr(self.config, "view_selection", "adaptive") == "fixed":
            elevs = list(self.config.candidate_camera_elevs)
            azims = list(self.config.candidate_camera_azims)
            weights = list(self.config.candidate_view_weights)
            if not (len(elevs) == len(azims) == len(weights) and len(elevs) > 0):
                raise ValueError("Camera elevations, azimuths and weights must have the same nonzero length")
        else:
            elevs, azims, weights = self.view_processor.bake_view_selection(
                self.config.candidate_camera_elevs, self.config.candidate_camera_azims,
                self.config.candidate_view_weights, self.config.max_selected_view_num,
            )
        if getattr(self.config, "coordinate_system", "materialmvp") == "miku":
            normals, positions, masks = self.view_processor.render_miku_multiview(elevs, azims)
        else:
            normals = self.view_processor.render_normal_multiview(elevs, azims, use_abs_coor=True)
            positions = self.view_processor.render_position_multiview(elevs, azims)
            masks = None
        conditions = dict(normal=normals, position=positions, masks=masks,
                          elevations=elevs, azimuths=azims, weights=weights)
        self.last_conditions = conditions
        debug_dir = getattr(self.config, "miku_debug_dir", None)
        if debug_dir:
            self.save_conditions(conditions, debug_dir, mesh_path)
        return conditions

    def save_conditions(self, conditions, directory, mesh_path=None):
        os.makedirs(directory, exist_ok=True)
        for kind in ("normal", "position", "masks"):
            for i, image in enumerate(conditions[kind] or []):
                image.save(os.path.join(directory, f"{kind}_{i:02d}.png"))
        metadata = {
            "mesh": mesh_path,
            "coordinate_system": getattr(self.config, "coordinate_system", "materialmvp"),
            "camera_type": getattr(self.config, "camera_type", "orth"),
            "camera_distance": self.render.camera_distance,
            "projection_matrix": np.asarray(self.render.camera_proj_mat).tolist(),
            "view_matrices": [self.render.get_view_matrix(e, a).tolist()
                              for e, a in zip(conditions["elevations"], conditions["azimuths"])],
            "elevations": conditions["elevations"], "azimuths": conditions["azimuths"],
            "weights": conditions["weights"],
            "render_size": self.config.render_size,
        }
        with open(os.path.join(directory, "cameras.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    @torch.no_grad()
    def predict_multiview(self, image_path, conditions, prompt=None):
        if isinstance(image_path, (str, os.PathLike)):
            image_prompt = [Image.open(image_path)]
        elif isinstance(image_path, Image.Image):
            image_prompt = [image_path]
        else:
            image_prompt = list(image_path)
        if not image_prompt:
            raise ValueError("At least one reference image is required")
        image_style = []
        is_miku = getattr(self.config, "multiview_backend", "materialmvp") == "miku"
        for image in image_prompt:
            if is_miku:
                image_style.append(image)  # Preserve alpha for reference crop/fit.
                continue
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, (255, 255, 255))
                white.paste(image, mask=image.getchannel("A"))
                image = white
            image_style.append(image.convert("RGB"))
        kwargs = {"masks": conditions["masks"]} if is_miku else {}
        return self.models["multiview_model"](
            image_style, conditions["normal"] + conditions["position"],
            prompt=prompt if prompt is not None else (None if is_miku else "high quality"),
            custom_view_size=self.config.resolution, resize_input=True, **kwargs,
        )

    @torch.no_grad()
    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None, use_remesh=False, save_glb=True,
                 prompt=None, mv_only=False):
        """Generate texture for 3D mesh using multiview diffusion"""
        path = os.path.dirname(os.fspath(mesh_path))
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, f"textured_mesh.obj")
        output_mesh_path = os.fspath(output_mesh_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
        conditions = self.prepare_conditions(mesh_path, use_remesh=use_remesh)
        multiviews_pbr = self.predict_multiview(image_path, conditions, prompt=prompt)
        if mv_only:
            return multiviews_pbr
        return self.bake_multiview(multiviews_pbr, conditions, output_mesh_path, save_glb=save_glb)

    @torch.no_grad()
    def bake_multiview(self, multiviews_pbr, conditions, output_mesh_path, save_glb=True):
        """Bake predictions against the mesh/cameras prepared on this instance."""
        selected_camera_elevs = conditions["elevations"]
        selected_camera_azims = conditions["azimuths"]
        selected_view_weights = conditions["weights"]
        output_mesh_path = os.fspath(output_mesh_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
        ###########  Enhance  ##########
        enhance_images = {}
        for channel, views in multiviews_pbr.items():
            if len(views) != len(selected_camera_elevs):
                raise ValueError(f"{channel}: prediction count does not match the rendering cameras")
            enhance_images[channel] = [image.resize((self.config.render_size, self.config.render_size),
                                                   Image.Resampling.LANCZOS) for image in views]
        texture, mask = self.view_processor.bake_from_multiview(
            enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
        )
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        ##########  inpaint  ###########
        texture = self.view_processor.texture_inpaint(texture, mask_np)
        self.render.set_texture(texture, force_set=True)
        if "mr" in enhance_images:
            texture_mr, mask_mr = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            self.render.set_texture_mr(
                texture_mr, force_set=getattr(self.config, "multiview_backend", "materialmvp") == "miku")

        is_miku = getattr(self.config, "multiview_backend", "materialmvp") == "miku"
        self.render.save_mesh(output_mesh_path, downsample=not is_miku)

        if save_glb:
            output_glb_path = os.path.splitext(output_mesh_path)[0] + ".glb"
            success = convert_obj_to_glb(output_mesh_path, output_glb_path,
                                         backend="trimesh" if is_miku else "blender")
            if not success:
                raise RuntimeError(f"Failed to export GLB: {output_glb_path}")

        return output_mesh_path
