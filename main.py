from pointcloud_processing import CleanParams, clean_and_downsample, visualize_point_cloud, save_point_cloud
from TIN_builder import build_TIN, save_TIN, visualize_TIN
from pathlib import Path
import numpy as np
from TIN_queries import TINQuerier

def run_default():
	input_path = 'point_cloud.ply'
	if not Path(input_path).exists():
		raise SystemExit(f'Missing {input_path} in workspace.')
	# Example: restrict to an XY window and optional Z span.
	# Set any bound to None to leave it open. Replace with real coordinates.
	bbox = ((None, None), (None, 0.6), (None, None))  # ((xmin,xmax),(ymin,ymax),(zmin,zmax)) None = unbound
	# Change mainly voxel size to downsize the points
	# Voxel size = area of cells in which points are reduces to a single instance
	# statistical neighbors = numbers of nearest neighbor points to measure distance to from each point
	# statistical ratio = threshold distance for mean neighbor distance over which points are considered outliers and thus erased.
	params = CleanParams(voxel_size_xy=0.01, statistical_nb_neighbors=10, statistical_std_ratio=0.01, bbox=bbox)
	cleaned = clean_and_downsample(input_path, params)
	print(f'Cleaned point cloud points: {len(cleaned.points)}')
	save_point_cloud(cleaned, 'cleaned_point_cloud.ply')
	print('Saved cleaned_point_cloud.ply')
 
	try:
		visualize_point_cloud(cleaned, 'Cleaned Point Cloud')
	except Exception as e:
		print(f'Visualization failed (maybe headless env): {e}')
  
	print('Building TIN...')
	V, F, Adjacency = build_TIN('cleaned_point_cloud.ply')
 
	save_TIN(V, F, Adjacency)
	print('Saved TIN data: TIN_vertices.npy, TIN_faces.npy, TIN_adjacency.npy')

	print(f'TIN vertex count: {len(V)}')
	print(f'TIN triangle count: {len(F)}')
	for i, neighbors in enumerate(Adjacency): 
		print(f'Face {i}: neighbors -> {neighbors}')

	visualize_TIN(V, F)

	# Iniciando Consultas sobre el TIN 

	try:
		V_loaded = np.load('TIN_vertices.npy')
		F_loaded = np.load('TIN_faces.npy')
	except FileNotFoundError:
		print("Archivos del TIN no encontrados. Saltando consultas.")
		return

	# 1. Crear una instancia del objeto de consultas
	querier = TINQuerier(V_loaded, F_loaded)

	# 2. Definir un punto de consulta (x, y)
	# Busca en el archivo cleaned_point_cloud.ply para encontrar un punto válido.
	# Se utiliza un punto de ejemplo aquí.
	query_point_xy = (0.5, 0.4) 

	# 3. Realizar las consultas
	face_idx, bary_coords = querier.find_triangle_for_point(query_point_xy)

	if face_idx is not None:
		print(f"Punto de consulta {query_point_xy} se encuentra en el triángulo con índice: {face_idx}")

		# Interpolar altura
		interpolated_z = querier.interpolate_height(query_point_xy)
		print(f" -> Altura (Z) interpolada en el punto: {interpolated_z:.4f}")

		# Obtener pendiente de la cara
		slope = querier.get_slope(face_idx)
		print(f" -> Pendiente de la cara del triángulo: {slope:.2f} grados")

		# Obtener orientación de la cara
		aspect = querier.get_aspect(face_idx)
		print(f" -> Orientación (Aspect) de la cara: {aspect:.2f} grados")
	else:
		print(f"El punto de consulta {query_point_xy} está fuera de los límites del TIN.")

  

if __name__ == '__main__':
	run_default()


