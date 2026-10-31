import trimesh
import xatlas


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    # Preserve source smooth normals across the UV seam vertex duplication.
    normals = mesh.vertex_normals.copy()
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.vertex_normals = normals[vmapping]
    mesh.visual.uv = uvs

    return mesh
