from pointcloud_processing import CleanParams, clean_and_downsample, visualize_point_cloud, save_point_cloud
from pathlib import Path

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

if __name__ == '__main__':
	run_default()

