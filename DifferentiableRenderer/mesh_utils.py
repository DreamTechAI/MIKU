import os
import cv2
import math
import numpy as np
from io import StringIO
from typing import Optional, Tuple, Dict, Any


def _safe_extract_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    """Extract nested attribute safely from object."""
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default


def _convert_to_numpy(data: Any, dtype: np.dtype) -> Optional[np.ndarray]:
    """Convert data to numpy array with specified dtype, handling None values."""
    if data is None:
        return None
    return np.asarray(data, dtype=dtype)


def load_mesh(mesh):
    """Load mesh data including vertices, faces, UV coordinates and texture."""
    # Extract vertex positions and face indices
    vtx_pos = _safe_extract_attribute(mesh, "vertices")
    pos_idx = _safe_extract_attribute(mesh, "faces")

    # Extract UV coordinates (reusing face indices for UV indices)
    vtx_uv = _safe_extract_attribute(mesh, "visual.uv")
    uv_idx = pos_idx  # Reuse face indices for UV mapping

    # Convert to numpy arrays with appropriate dtypes
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)

    texture_data = None
    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data


def _get_base_path_and_name(mesh_path: str) -> Tuple[str, str]:
    """Get base path without extension and mesh name."""
    base_path = os.path.splitext(mesh_path)[0]
    name = os.path.basename(base_path)
    return base_path, name


def _save_texture_map(
    texture: np.ndarray,
    base_path: str,
    suffix: str = "",
    image_format: str = ".jpg",
    color_convert: Optional[int] = None,
) -> str:
    """Save texture map with optional color conversion."""
    path = f"{base_path}{suffix}{image_format}"
    if image_format.lower() == ".png":
        if not np.isfinite(texture).all():
            raise ValueError(f"Nonfinite texture cannot be exported: {path}")
        processed_texture = np.rint(np.clip(texture, 0, 1) * 255).astype(np.uint8)
    else:
        processed_texture = (texture * 255).astype(np.uint8)

    if color_convert is not None:
        processed_texture = cv2.cvtColor(processed_texture, color_convert)
        cv2.imwrite(path, processed_texture)
    else:
        cv2.imwrite(path, processed_texture[..., ::-1])  # RGB to BGR

    return os.path.basename(path)


def _write_mtl_properties(f, properties: Dict[str, Any]):
    """Write material properties to MTL file."""
    for key, value in properties.items():
        if isinstance(value, (list, tuple)):
            f.write(f"{key} {' '.join(map(str, value))}\n")
        else:
            f.write(f"{key} {value}\n")


def _create_obj_content(
    vtx_pos: np.ndarray, vtx_uv: np.ndarray, pos_idx: np.ndarray, uv_idx: np.ndarray, name: str
) -> str:
    """Create OBJ file content."""
    buffer = StringIO()

    # Write header and vertices
    buffer.write(f"mtllib {name}.mtl\no {name}\n")
    np.savetxt(buffer, vtx_pos, fmt="v %.6f %.6f %.6f")
    np.savetxt(buffer, vtx_uv, fmt="vt %.6f %.6f")
    buffer.write("s 0\nusemtl Material\n")

    # Write faces
    pos_idx_plus1 = pos_idx + 1
    uv_idx_plus1 = uv_idx + 1
    face_format = np.frompyfunc(lambda *x: f"{int(x[0])}/{int(x[1])}", 2, 1)
    faces = face_format(pos_idx_plus1, uv_idx_plus1)
    face_strings = [f"f {' '.join(face)}" for face in faces]
    buffer.write("\n".join(face_strings) + "\n")

    return buffer.getvalue()


def save_obj_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None,
                  image_format=".jpg"):
    """Save mesh as OBJ file with textures and material."""
    # Convert inputs to numpy arrays
    vtx_pos = _convert_to_numpy(vtx_pos, np.float32)
    vtx_uv = _convert_to_numpy(vtx_uv, np.float32)
    pos_idx = _convert_to_numpy(pos_idx, np.int32)
    uv_idx = _convert_to_numpy(uv_idx, np.int32)

    base_path, name = _get_base_path_and_name(mesh_path)

    # Create and save OBJ content
    obj_content = _create_obj_content(vtx_pos, vtx_uv, pos_idx, uv_idx, name)
    with open(mesh_path, "w") as obj_file:
        obj_file.write(obj_content)

    # Save texture maps
    texture_maps = {}
    texture_maps["diffuse"] = _save_texture_map(texture, base_path, image_format=image_format)

    if metallic is not None:
        texture_maps["metallic"] = _save_texture_map(
            metallic, base_path, "_metallic", image_format=image_format, color_convert=cv2.COLOR_RGB2GRAY
        )
    if roughness is not None:
        texture_maps["roughness"] = _save_texture_map(
            roughness, base_path, "_roughness", image_format=image_format, color_convert=cv2.COLOR_RGB2GRAY
        )
    if normal is not None:
        texture_maps["normal"] = _save_texture_map(normal, base_path, "_normal", image_format=image_format)

    # Create MTL file
    _create_mtl_file(base_path, texture_maps, metallic is not None)


def _create_mtl_file(base_path: str, texture_maps: Dict[str, str], is_pbr: bool):
    """Create MTL material file."""
    mtl_path = f"{base_path}.mtl"

    with open(mtl_path, "w") as f:
        f.write("newmtl Material\n")

        if is_pbr:
            # PBR material properties
            properties = {
                "Kd": [0.800, 0.800, 0.800],
                "Ke": [0.000, 0.000, 0.000],  # 鐜鍏夐伄钄�
                "Ni": 1.500,  # 鎶樺皠绯绘暟
                "d": 1.0,  # 閫忔槑搴�
                "illum": 2,  # 鍏夌収妯″瀷
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)

            # Additional PBR maps
            map_configs = [("metallic", "map_Pm"), ("roughness", "map_Pr"), ("normal", "map_Bump -bm 1.0")]

            for texture_key, mtl_key in map_configs:
                if texture_key in texture_maps:
                    f.write(f"{mtl_key} {texture_maps[texture_key]}\n")
        else:
            # Standard material properties
            properties = {
                "Ns": 250.000000,
                "Ka": [0.200, 0.200, 0.200],
                "Kd": [0.800, 0.800, 0.800],
                "Ks": [0.500, 0.500, 0.500],
                "Ke": [0.000, 0.000, 0.000],
                "Ni": 1.500,
                "d": 1.0,
                "illum": 3,
                "map_Kd": texture_maps["diffuse"],
            }
            _write_mtl_properties(f, properties)


def save_mesh(mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=None, roughness=None, normal=None,
              image_format=".jpg"):
    """Save mesh using OBJ format."""
    save_obj_mesh(
        mesh_path, vtx_pos, pos_idx, vtx_uv, uv_idx, texture, metallic=metallic, roughness=roughness, normal=normal,
        image_format=image_format,
    )


def _load_metallic_roughness_texture(obj_path):
    """Pack this OBJ's declared MTL maps as linear glTF G=roughness, B=metallic.

    Read the MTL references instead of guessing sidecar names: an albedo-only
    rerun may leave older MR images beside the OBJ, and must not reuse them.
    """
    from pathlib import Path
    from PIL import Image

    obj_path = Path(obj_path)
    libraries = []
    for line in obj_path.read_text().splitlines():
        key, _, value = line.strip().partition(" ")
        if key == "mtllib":
            libraries.append(obj_path.parent / value.strip())
    maps = {}
    for library in libraries:
        for line in library.read_text().splitlines():
            key, _, value = line.strip().partition(" ")
            if key in ("map_Pm", "map_Pr"):
                maps[key] = library.parent / value.strip()
    if not maps:
        return None
    if set(maps) != {"map_Pm", "map_Pr"}:
        raise ValueError("Both metallic and roughness maps are required for PBR GLB export")
    with Image.open(maps["map_Pm"]) as source:
        metallic = source.convert("L")
    with Image.open(maps["map_Pr"]) as source:
        roughness = source.convert("L")
    if metallic.size != roughness.size:
        raise ValueError("Metallic and roughness textures must have identical dimensions")
    # The red component is unused by metallicRoughnessTexture. Do not attach an
    # occlusionTexture: this model's red prediction is not validated AO.
    packed = Image.merge("RGB", (Image.new("L", metallic.size, 255), roughness, metallic))
    packed.save(obj_path.with_name(obj_path.stem + "_metallic_roughness.png"))
    return packed


def convert_obj_to_glb(
    obj_path: str,
    glb_path: str,
    shade_type: str = "SMOOTH",
    auto_smooth_angle: float = 60,
    merge_vertices: bool = False,
    backend: str = "trimesh",
) -> bool:
    """Convert OBJ file to GLB with explicit lossless PBR textures."""
    if backend == "trimesh":
        import trimesh
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        image = getattr(mesh.visual.material, "image", None)
        if image is None:
            raise ValueError(f"No base-color texture found in {obj_path}")
        metallic_roughness = _load_metallic_roughness_texture(obj_path)
        mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=image,
            metallicRoughnessTexture=metallic_roughness,
            metallicFactor=1.0 if metallic_roughness is not None else 0.0,
            roughnessFactor=1.0,
        )
        mesh.export(glb_path, file_type="glb")
        return True
    raise ValueError("Only the trimesh GLB export backend is included")
