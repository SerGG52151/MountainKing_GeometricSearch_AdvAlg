import open3d as o3d
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Union

@dataclass
class CleanParams:
    z_min: Optional[float] = None  # If provided, clip below
    z_max: Optional[float] = None  # If provided, clip above
    statistical_nb_neighbors: int = 20
    statistical_std_ratio: float = 2.0
    voxel_size_xy: float = 1.0  # horizontal resolution (meters)
    # Axis-aligned crop bounding box: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) each item can be None for open bound
    bbox: Optional[Tuple[Tuple[Optional[float], Optional[float]],
                         Tuple[Optional[float], Optional[float]],
                         Tuple[Optional[float], Optional[float]]]] = None


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    ext = Path(path).suffix.lower()
    if ext == '.ply':
        pcd = o3d.io.read_point_cloud(path)
    elif ext == '.pcd':
        pcd = o3d.io.read_point_cloud(path)
    elif ext == '.csv':
        pts = np.loadtxt(path, delimiter=',', usecols=(0,1,2), dtype=float)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    elif ext == '.xyz':
        pts = np.loadtxt(path, dtype=float, usecols=(0,1,2))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    else:
        raise ValueError(f'Unsupported file extension: {ext}')
    if len(pcd.points) == 0:
        raise ValueError('Loaded point cloud is empty.')
    return pcd


def clip_z(pcd: o3d.geometry.PointCloud, z_min: Optional[float], z_max: Optional[float]) -> o3d.geometry.PointCloud:
    if z_min is None and z_max is None:
        return pcd
    pts = np.asarray(pcd.points)
    mask = np.ones(len(pts), dtype=bool)
    if z_min is not None:
        mask &= pts[:,2] >= z_min
    if z_max is not None:
        mask &= pts[:,2] <= z_max
    filtered = pts[mask]
    out = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered))
    return out


def statistical_outlier_removal(pcd: o3d.geometry.PointCloud, nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    if len(pcd.points) < nb_neighbors + 2:
        return pcd  # not enough points, skip
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl


def voxel_downsample_xy(pcd: o3d.geometry.PointCloud, voxel_size_xy: float) -> o3d.geometry.PointCloud:
    # We want XY voxel while preserving average Z inside each XY cell.
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return pcd
    if voxel_size_xy <= 0:
        return pcd
    # Compute voxel indices in XY only
    xy = pts[:, :2]
    mins = xy.min(axis=0)
    ij = np.floor((xy - mins) / voxel_size_xy).astype(int)
    # Hash 2D indices
    hashes = ij[:,0].astype(np.int64) << 32 | (ij[:,1].astype(np.int64) & 0xffffffff)
    # Aggregate using dict of lists for mean
    from collections import defaultdict
    accum_x = defaultdict(list)
    accum_y = defaultdict(list)
    accum_z = defaultdict(list)
    for h, (x,y,z) in zip(hashes, pts):
        accum_x[h].append(x)
        accum_y[h].append(y)
        accum_z[h].append(z)
    new_pts = []
    for h in accum_x.keys():
        new_pts.append([
            np.mean(accum_x[h]),
            np.mean(accum_y[h]),
            np.mean(accum_z[h])
        ])
    new_pts = np.array(new_pts)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(new_pts))


BBoxType = Optional[Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]]


def _apply_bbox(pcd: o3d.geometry.PointCloud, bbox: BBoxType) -> o3d.geometry.PointCloud:
    if bbox is None:
        return pcd
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
    pts = np.asarray(pcd.points)
    mask = np.ones(len(pts), dtype=bool)
    if xmin is not None:
        mask &= pts[:,0] >= xmin
    if xmax is not None:
        mask &= pts[:,0] <= xmax
    if ymin is not None:
        mask &= pts[:,1] >= ymin
    if ymax is not None:
        mask &= pts[:,1] <= ymax
    if zmin is not None:
        mask &= pts[:,2] >= zmin
    if zmax is not None:
        mask &= pts[:,2] <= zmax
    if mask.all():
        return pcd
    filtered = pts[mask]
    new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered))
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if len(colors) == len(pts):
            new_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return new_pcd


def clean_and_downsample(path: str, params: CleanParams) -> o3d.geometry.PointCloud:
    pcd = load_point_cloud(path)
    # Optional global z clip first (legacy)
    pcd = clip_z(pcd, params.z_min, params.z_max)
    # Axis aligned bbox (may also constrain z or override z_min/z_max if both used)
    pcd = _apply_bbox(pcd, params.bbox)
    # XY voxel aggregation FIRST (thin before statistics)
    pcd = voxel_downsample_xy(pcd, params.voxel_size_xy)
    # Statistical outlier removal AFTER thinning
    if params.statistical_nb_neighbors > 1 and len(pcd.points) > 3:
        # Adjust k to be valid relative to current number of points
        k_eff = min(params.statistical_nb_neighbors, max(2, len(pcd.points) - 1))
        if k_eff >= 2:
            try:
                pcd = statistical_outlier_removal(pcd, k_eff, params.statistical_std_ratio)
            except Exception as e:
                print(f"Statistical outlier removal skipped: {e}")
    return pcd


def visualize_point_cloud(pcd: o3d.geometry.PointCloud, window_name: str = 'PointCloud'):
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def save_point_cloud(pcd: o3d.geometry.PointCloud, path: str):
    ext = Path(path).suffix.lower()
    if ext == '.ply':
        o3d.io.write_point_cloud(path, pcd)
    elif ext in ('.xyz', '.csv'):
        pts = np.asarray(pcd.points)
        if ext == '.csv':
            np.savetxt(path, pts, delimiter=',')
        else:
            np.savetxt(path, pts)
    else:
        raise ValueError('Unsupported export format (use .ply, .xyz, or .csv).')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Clean and downsample point cloud.')
    parser.add_argument('--input', required=True, help='Input point cloud (.ply/.pcd/.csv/.xyz)')
    parser.add_argument('--output', required=False, help='Output cleaned point cloud path')
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--nb-neighbors', type=int, default=20)
    parser.add_argument('--std-ratio', type=float, default=2.0)
    parser.add_argument('--voxel', type=float, default=1.0, help='XY voxel size (meters)')
    parser.add_argument('--no-view', action='store_true', help='Skip visualization')
    # Bounding box arguments (optional)
    parser.add_argument('--xmin', type=float, default=None)
    parser.add_argument('--xmax', type=float, default=None)
    parser.add_argument('--ymin', type=float, default=None)
    parser.add_argument('--ymax', type=float, default=None)
    parser.add_argument('--crop-zmin', type=float, default=None, help='Bounding box z min (distinct from --z-min which applies earlier)')
    parser.add_argument('--crop-zmax', type=float, default=None, help='Bounding box z max (distinct from --z-max which applies earlier)')
    args = parser.parse_args()

    bbox = None
    if any(v is not None for v in (args.xmin, args.xmax, args.ymin, args.ymax, args.crop_zmin, args.crop_zmax)):
        bbox = (
            (args.xmin, args.xmax),
            (args.ymin, args.ymax),
            (args.crop_zmin, args.crop_zmax)
        )
    params = CleanParams(
        z_min=args.z_min,
        z_max=args.z_max,
        statistical_nb_neighbors=args.nb_neighbors,
        statistical_std_ratio=args.std_ratio,
        voxel_size_xy=args.voxel,
        bbox=bbox
    )
    cleaned = clean_and_downsample(args.input, params)

    if args.output:
        save_point_cloud(cleaned, args.output)
        print(f'Saved cleaned point cloud to {args.output} ({len(cleaned.points)} pts)')
    else:
        print(f'Cleaned point cloud has {len(cleaned.points)} points (no output path provided)')

    if not args.no_view:
        visualize_point_cloud(cleaned, 'Cleaned Point Cloud')
