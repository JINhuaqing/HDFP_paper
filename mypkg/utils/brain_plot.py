# some fns for plotting vector on brain
# They are used with the nilearn pkg
# all from https://github.com/CarloNicolini/brainroisurf/blob/master/brainroisurf/nilearndrawmembership.py
import numpy as np
def normalize_v3(arr):
    """
    Normalize a numpy array of 3 component vectors shape=(n,3)
    :param arr: numpy array of shape (n, 3)
    :return: normalized numpy array of shape (n, 3)
    """
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)

    # hack
    lens[lens == 0.0] = 1.0

    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


def compute_normals(vertices, triangles):
    """
    Compute the normals for each vertex in a mesh given its vertices and triangles
    :param vertices: numpy array of shape (n, 3) representing the vertices of the mesh
    :param triangles: numpy array of shape (m, 3) representing the triangles of the mesh
    :return: numpy array of shape (n, 3) representing the normals for each vertex
    """
    # A normal is a vector that is perpendicular to the surface of a 3D object.
    # In the context of this function, we are calculating the normals for each vertex in a mesh.

    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[triangles]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[triangles[:, 0]] += n
    norm[triangles[:, 1]] += n
    norm[triangles[:, 2]] += n
    normalize_v3(norm)

    return norm

def get_bg_data(XYZs, faces):
    """
    Calculate the background data for a mesh given its vertices and faces
    :param XYZs: numpy array of shape (n, 3) representing the vertices of the mesh
    :param faces: numpy array of shape (m, 3) representing the faces of the mesh
    :return: numpy array of shape (n,) representing the background data for each vertex
    """
    normals = compute_normals(XYZs, faces)
    l = np.max(XYZs) # position of light as a scalar
    light_pos = np.array([0, 0, 4*l])
    light_dir = XYZs - light_pos  # use broadcasting, vertices - light
    light_intensity = np.einsum('ij,ij->i', light_dir, normals)
    bg_data = light_intensity**2
    return bg_data