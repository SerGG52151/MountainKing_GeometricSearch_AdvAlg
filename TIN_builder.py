import numpy as np
from scipy.spatial import Delaunay, cKDTree
import open3d as o3d

def build_TIN(point_cloud_path):
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    
    if points.shape[1] != 3:
        raise ValueError("Point cloud must be 3D.")
    
    xy = points[:, :2]
    
    delaunay = Delaunay(xy)
    tri_indices = delaunay.simplices 
    
    unique_xy, inverse_indices = np.unique(xy, axis=0, return_inverse=True)
    
    kdtree = cKDTree(points[:, :2])
    _, idx = kdtree.query(unique_xy)
    z_values = points[idx, 2]
    
    V = np.column_stack((unique_xy, z_values))
    F = inverse_indices[tri_indices]
    
    neighbors = [set() for _ in range(len(F))]
    edge_to_face = {}
    
    for i, face in enumerate(F):
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for a, b in edges:
            edge = tuple(sorted((a, b)))
            if edge in edge_to_face:
                neighbor_face = edge_to_face[edge]
                neighbors[i].add(neighbor_face)
                neighbors[neighbor_face].add(i)
            else:
                edge_to_face[edge] = i

    Adjacency = [sorted(list(n)) for n in neighbors]

    return V, F, Adjacency

def save_TIN(V, F, Adjacency, prefix='TIN'):
    np.save(f'{prefix}_vertices.npy', V)
    np.save(f'{prefix}_faces.npy', F)
    np.save(f'{prefix}_adjacency.npy', np.array(Adjacency, dtype=object))
    print(f'TIN saved as {prefix}_*.npy files')
    
    
def visualize_TIN(V, F):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)

    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [mesh],
        window_name='TIN Mesh',
        mesh_show_back_face=True
    )