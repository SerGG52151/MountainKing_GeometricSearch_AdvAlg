# main.py
from pathlib import Path
import numpy as np

from pointcloud_processing import (
    CleanParams,
    clean_and_downsample,
    visualize_point_cloud,
    save_point_cloud,
)
from TIN_builder import build_TIN, save_TIN, visualize_TIN
from TIN_queries import TINQuerier  # asegúrate de usar la versión extendida

# ------------------------ Config rápidos ------------------------
INPUT_PATH = "point_cloud.ply"
# BBox: ((xmin, xmax), (ymin, ymax), (zmin, zmax)) ; None = sin límite
BBOX = ((None, None), (None, 0.6), (None, None))

# Parámetros de limpieza (valores más "sensatos" que los de demo)
VOXEL = 0.03                 # ~3 cm; sube si tienes demasiados puntos
NB_NEIGH = 30                # vecinos para filtro estadístico
STD_RATIO = 1.5              # umbral en desviaciones estándar (1.0–2.0 típico)

# Ejemplos (punto 4)
SLOPE_PRUNE = 25.0           # poda aristas con pendiente media > 25° (None para no podar)
CONTOUR_STEP = None          # e.g., 2.0 para cada 2 m; None = solo nivel mediano


def run_default():
    # --------------------- 1) Clean + downsample ---------------------
    if not Path(INPUT_PATH).exists():
        raise SystemExit(f"Missing {INPUT_PATH} in workspace.")

    params = CleanParams(
        voxel_size_xy=VOXEL,
        statistical_nb_neighbors=NB_NEIGH,
        statistical_std_ratio=STD_RATIO,
        bbox=BBOX,
    )

    cleaned = clean_and_downsample(INPUT_PATH, params)
    print(f"[Clean] Puntos tras limpieza/downsample: {len(cleaned.points)}")
    save_point_cloud(cleaned, "cleaned_point_cloud.ply")
    print("[Clean] Guardado: cleaned_point_cloud.ply")

    try:
        visualize_point_cloud(cleaned, "Cleaned Point Cloud")
    except Exception as e:
        print(f"(Aviso) Visualización de nube omitida: {e}")

    # ------------------------- 2) Build 2.5D TIN -------------------------
    print("[TIN] Construyendo TIN...")
    try:
        V, F, Adjacency = build_TIN("cleaned_point_cloud.ply")
    except Exception as e:
        raise SystemExit(f"[TIN] Error al construir TIN: {e}")

    save_TIN(V, F, Adjacency)
    print("[TIN] Guardado: TIN_vertices.npy, TIN_faces.npy, TIN_adjacency.npy")
    print(f"[TIN] Vértices: {len(V)} | Triángulos: {len(F)}")

    # (Opcional) imprimir vecindades (cuidado si M es grande)
    if len(F) <= 200:
        for i, neighbors in enumerate(Adjacency):
            print(f"  - Cara {i}: vecinos -> {neighbors}")

    try:
        visualize_TIN(V, F)
    except Exception as e:
        print(f"(Aviso) Visualización de TIN omitida: {e}")

    # --------------------- 3) Tiny utilities / consultas ---------------------
    try:
        V_loaded = np.load("TIN_vertices.npy")
        F_loaded = np.load("TIN_faces.npy")
    except FileNotFoundError:
        print("[Query] Archivos del TIN no encontrados. Saltando consultas.")
        return

    # Instanciar el consultor (con grid uniforme para acelerar)
    querier = TINQuerier(V_loaded, F_loaded)

    # --- Punto de prueba seguro: centroide del triángulo 0 (garantiza 'inside') ---
    if len(F_loaded) == 0:
        print("[Query] Malla vacía; no hay consultas que ejecutar.")
        return
    tri0 = F_loaded[0]
    centroid_xy = tuple(np.mean(V_loaded[tri0, :2], axis=0))

    print("\n=== DEMOS de búsquedas geométricas ===")

    # (1) Elevation at (x,y) + slope/aspect
    xy = centroid_xy
    z = querier.interpolate_height(xy)
    fi, _ = querier.find_triangle_for_point(xy)
    print(f"[1] Elevation en {xy}: z = {z:.4f}")
    if fi is not None:
        print(f"    slope = {querier.get_slope(fi):.2f}°, aspect = {querier.get_aspect(fi):.2f}°")
    else:
        print("    (No se encontró triángulo contenedor; raro con el centroide).")

    # (2) Nearest-surface point para q=(x,y,z)
    q_xyz = (xy[0], xy[1], z + 5.0)  # 5 unidades por encima
    res = querier.nearest_surface_point(q_xyz)
    print(f"[2] Nearest-surface de {q_xyz}:")
    print(f"    foot = {res['foot']}, |dz| = {res['vertical_distance']:.4f}, where = {res['where']}")

    # (3) Shortest path (hiking) con poda opcional por pendiente
    # Elegimos dos vértices 'lejanos' aproximando por extremos de (x+y)
    start_idx = int(np.argmin(V_loaded[:, 0] + V_loaded[:, 1]))
    goal_idx  = int(np.argmax(V_loaded[:, 0] + V_loaded[:, 1]))
    start_xy = tuple(V_loaded[start_idx, :2])
    goal_xy  = tuple(V_loaded[goal_idx, :2])

    path_idx, path_xyz, total_len = querier.shortest_path(
        start_xy=start_xy, goal_xy=goal_xy, slope_deg_max=SLOPE_PRUNE
    )
    if len(path_idx) == 0:
        print("[3] No se encontró camino (quizá poda demasiado estricta).")
    else:
        print(f"[3] Shortest path: {len(path_idx)} nodos; longitud 3D = {total_len:.3f}")

    # (4) Contours (isohipsas)
    zmin, zmax = float(V_loaded[:, 2].min()), float(V_loaded[:, 2].max())
    if CONTOUR_STEP is None:
        levels = [float(np.median(V_loaded[:, 2]))]
    else:
        levels = list(np.arange(np.floor(zmin), np.ceil(zmax) + 1e-9, float(CONTOUR_STEP)))

    conts = querier.contours(levels)
    for c in levels:
        n_polys = len(conts[c])
        n_pts = sum(len(poly) for poly in conts[c])
        print(f"[4] Contours z={c:.3f}: {n_polys} polilíneas, {n_pts} vértices en total")
    try:
        import open3d as o3d
        if levels:
            c = levels[0]
            geoms = []
            for poly in conts[c]:
                if len(poly) < 2:
                    continue
                points = o3d.utility.Vector3dVector(poly)
                lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(poly) - 1)])
                ls = o3d.geometry.LineSet(points=points, lines=lines)
                geoms.append(ls)
            if geoms:
                o3d.visualization.draw_geometries(geoms, window_name=f"Contours z={c}")
    except Exception as e:
        print(f"(Aviso) Visualización de contornos omitida: {e}")


if __name__ == "__main__":
    run_default()
