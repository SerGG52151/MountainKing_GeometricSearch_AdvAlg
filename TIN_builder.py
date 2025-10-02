import numpy as np
from scipy.spatial import Delaunay, cKDTree
import open3d as o3d

def build_TIN(point_cloud_path):
    # descarga la nube de puntos, y los puntos se guardan como un arreglo de numpy
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    
    if points.shape[1] != 3:
        raise ValueError("Point cloud must be 3D.")
    
    #extraer unicamente los puntos en el plano x-y
    xy = points[:, :2]
    
    #triangulacion delaunay
    delaunay = Delaunay(xy)
    #lista de vertices de triangulos obtenidos por Delaunay
    tri_indices = delaunay.simplices 
    
    #elimina coordinadas xy duplicadas
    #mapea coordinadas originales a su indice en unique_xy
    unique_xy, inverse_indices = np.unique(xy, axis=0, return_inverse=True)
    
    #Usa KDTree para encontrar el punto 3D mas cercano para cada unique_xy
    kdtree = cKDTree(points[:, :2])
    _, idx = kdtree.query(unique_xy)
    #Sacamos la coordinada del eje-Z de la nube original
    z_values = points[idx, 2]
    
    # V = lista de vertice 3D que empareja cada unique_xy con su posicion Z.
    # F = Caras/triangulos
    V = np.column_stack((unique_xy, z_values))
    F = inverse_indices[tri_indices]
    
    #Inicializa estructura de adyacencia
    neighbors = [set() for _ in range(len(F))]
    edge_to_face = {}
    
    #Para cada triangulo, checa las orillas
    for i, face in enumerate(F):
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for a, b in edges:
            edge = tuple(sorted((a, b)))
            #Si las orillas se han visto previamente, registra ambas orillas como vecinos
            if edge in edge_to_face:
                neighbor_face = edge_to_face[edge]
                neighbors[i].add(neighbor_face)
                neighbors[neighbor_face].add(i)
            else:
                edge_to_face[edge] = i

    #Sortea la lista de adyacencia
    Adjacency = [sorted(list(n)) for n in neighbors]

    return V, F, Adjacency

# Guarda los valores de V, F, y la lista de Adyacencia en archivos .npy para reutilizar
def save_TIN(V, F, Adjacency, prefix='TIN'):
    np.save(f'{prefix}_vertices.npy', V)
    np.save(f'{prefix}_faces.npy', F)
    np.save(f'{prefix}_adjacency.npy', np.array(Adjacency, dtype=object))
    print(f'TIN saved as {prefix}_*.npy files')
    
# Funcion para visualizar la mesh de la nube de puntos
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