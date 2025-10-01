# Mountain Point Cloud Processing

This repository processes an unstructured mountain point cloud to prepare for building a 2.5D TIN and geometric search queries.

## Step 1: Clean + Downsample
Input: raw (x,y,z) points from photogrammetry (`.ply`, `.pcd`, `.csv`, `.xyz`).

Operations performed:
1. Optional Z clipping (remove points below/above thresholds)
2. Statistical outlier removal (Open3D) with configurable neighbor count and std ratio
3. XY voxel (grid) downsample while averaging Z (preserves terrain surface trend)

### Installation

Python 3.10 or 3.11 recommended (Open3D prebuilt wheels support these well).

Create and activate a virtual environment (Windows PowerShell):

```
python -m venv .venv
./.venv/Scripts/Activate
pip install --upgrade pip
pip install -r requirements.txt
```

Mac/Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

Activate your virtual environment if not already active, then:

```
python pointcloud_processing.py --input point_cloud.ply --output cleaned.ply --voxel 1.0 --nb-neighbors 24 --std-ratio 2.0
```

Optional arguments:
- `--z-min <value>` / `--z-max <value>`: pre-clip extreme elevations (applied before bbox)
- `--xmin/xmax/ymin/ymax/crop-zmin/crop-zmax`: axis-aligned bounding box crop
- `--voxel <meters>`: XY grid size (0.5–2.0 typical). Larger = more aggressive reduction.
- `--nb-neighbors`: K for statistical outlier removal (after voxel thinning)
- `--std-ratio`: threshold multiplier; higher keeps more points
- `--no-view`: skip visualization window (e.g., for headless runs)

### Output
A cleaned and downsampled point set ready for TIN generation.

Next steps (planned):
- Step 2: Generate 2.5D TIN via 2D Delaunay on (x,y) with retained z
- Step 3: Build spatial indices (triangle bounding boxes / k-d tree for vertices)
- Step 4: Implement queries (height at (x,y), slope/aspect, nearest point, line-of-sight skeleton)

---

## Development Notes
- The XY voxel downsample groups by XY cell and averages (x,y,z); you could switch to selecting lowest z if focusing on ground vs vegetation.
- For dense vegetation or noise, consider radius outlier removal as an additional step.
- Current ordering: load -> (z clip) -> (bbox) -> voxel thin -> statistical outlier removal.
- Adjust neighbor count down if voxel thinning leaves very few points.
