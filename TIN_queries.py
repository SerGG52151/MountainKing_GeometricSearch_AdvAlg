# --- Extensión TINQuerier: consultas geométricas rápidas y simples ---
import numpy as np
from collections import defaultdict, deque
from heapq import heappush, heappop
from scipy.spatial import cKDTree

class TINQuerier:
    def __init__(self, V, F, build_grid=True, grid_cell_size=None):
        if V.shape[1] != 3 or F.shape[1] != 3:
            raise ValueError("Vértices (V) y Caras (F) deben tener 3 columnas.")

        self.V = V
        self.F = F

        # Normales por cara (vectorizadas)
        v0 = self.V[self.F[:, 0], :]
        v1 = self.V[self.F[:, 1], :]
        v2 = self.V[self.F[:, 2], :]
        self.normals = np.cross(v1 - v0, v2 - v0)
        self.xy = self.V[:, :2]
        self.z  = self.V[:, 2]

        # KDTree 2D sobre vértices (para nearest-vertex)
        self._kdtree_xy = cKDTree(self.xy)

        # Incidencias vértice-cara y aristas
        self.vert_to_faces = [set() for _ in range(len(self.V))]
        self.edge_to_faces = defaultdict(list)
        for fi, (a, b, c) in enumerate(self.F):
            self.vert_to_faces[a].add(fi)
            self.vert_to_faces[b].add(fi)
            self.vert_to_faces[c].add(fi)
            for u, v in ((a, b), (b, c), (c, a)):
                e = (u, v) if u < v else (v, u)
                self.edge_to_faces[e].append(fi)

        # Índice uniforme (grid) sobre XY para filtrar candidatos
        self.grid = None
        if build_grid:
            self._build_uniform_grid(grid_cell_size)

    # ------------------- Infraestructura: grid uniforme -------------------
    def _build_uniform_grid(self, cell_size=None):
        Vxy = self.xy
        F = self.F

        # Tamaño de celda: por defecto ~ 2-3 veces la mediana de longitudes de arista en XY
        if cell_size is None:
            # Muestra rápida de aristas para estimar mediana
            edges = set()
            for (a, b, c) in F:
                for u, v in ((a, b), (b, c), (c, a)):
                    if u > v: u, v = v, u
                    edges.add((u, v))
            if not edges:
                raise ValueError("Malla sin aristas.")
            elens = [np.linalg.norm(Vxy[u] - Vxy[v]) for (u, v) in list(edges)[: min(len(edges), 5000)]]
            med = np.median(elens) if len(elens) else 1.0
            cell_size = max(1e-9, 2.5 * med)

        self.cell_size = float(cell_size)
        mins = Vxy.min(axis=0) - 1e-9
        maxs = Vxy.max(axis=0) + 1e-9
        self.grid_min = mins
        self.grid_shape = np.ceil((maxs - mins) / self.cell_size).astype(int)

        grid = defaultdict(list)
        for fi, face in enumerate(F):
            pts = Vxy[np.array(face)]
            bbmin = np.floor((pts.min(axis=0) - self.grid_min) / self.cell_size).astype(int)
            bbmax = np.floor((pts.max(axis=0) - self.grid_min) / self.cell_size).astype(int)
            for ix in range(bbmin[0], bbmax[0] + 1):
                for iy in range(bbmin[1], bbmax[1] + 1):
                    grid[(ix, iy)].append(fi)
        self.grid = grid

    def _cell_of(self, p_xy):
        ij = np.floor((np.asarray(p_xy) - self.grid_min) / self.cell_size).astype(int)
        return (int(ij[0]), int(ij[1]))

    def _candidate_faces(self, p_xy):
        """Devuelve candidatos desde la celda (y anillos vecinos si necesario)."""
        if self.grid is None:
            return range(len(self.F))
        ij = self._cell_of(p_xy)
        cand = self.grid.get(ij, [])
        if cand:
            return cand
        # Expandir a vecinos si está vacío (hasta 5 anillos)
        for r in range(1, 6):
            cands = []
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    cands.extend(self.grid.get((ij[0] + dx, ij[1] + dy), []))
            if cands:
                return cands
        # Fallback
        return range(len(self.F))

    # ------------------- (1) Elevation at query (x,y) -------------------
    def find_triangle_for_point(self, p_xy):
        """Usa baricéntricas en XY; ahora filtra por candidatos del grid."""
        p = np.array(p_xy, dtype=float)
        for i in self._candidate_faces(p):
            face = self.F[i]
            v0 = self.xy[face[0]]
            v1 = self.xy[face[1]]
            v2 = self.xy[face[2]]

            v10 = v1 - v0; v20 = v2 - v0; vp0 = p - v0
            d00 = np.dot(v10, v10); d01 = np.dot(v10, v20); d11 = np.dot(v20, v20)
            d20 = np.dot(vp0, v10); d21 = np.dot(vp0, v20)
            denom = d00 * d11 - d01 * d01
            if np.abs(denom) < 1e-12:
                continue
            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            w = 1.0 - u - v
            if u >= -1e-9 and v >= -1e-9 and w >= -1e-9:
                return i, np.array([w, u, v])
        return None, None

    def interpolate_height(self, p_xy):
        fi, bary = self.find_triangle_for_point(p_xy)
        if fi is None:
            return None
        vz = self.z[self.F[fi]]
        return float(np.dot(bary, vz))

    def get_slope(self, face_index):
        nx, ny, nz = self.normals[face_index]
        if np.abs(nz) < 1e-12:
            return 90.0
        slope_rad = np.arctan(np.sqrt(nx * nx + ny * ny) / np.abs(nz))
        return float(np.degrees(slope_rad))

    def get_aspect(self, face_index):
        nx, ny, nz = self.normals[face_index]
        if np.abs(nz) < 1e-12:
            return np.nan
        # Aspecto GIS: dirección de máxima bajada, 0°=N, 90°=E
        dx = nx / nz; dy = ny / nz
        theta = np.degrees(np.arctan2(dy, dx))   # desde +X CCW
        return float((450.0 - theta) % 360.0)

    # ------------------- (2) Nearest-surface point for 3D q -------------------
    def nearest_surface_point(self, q_xyz):
        """
        Proyecta q a XY. Si cae en un triángulo, devuelve pie sobre la superficie.
        Si no, 'clamp' al vértice o arista más cercana (en XY) del 1-ring del vértice más cercano.
        Return: dict con foot(x,y,z), vertical_distance, where='inside|edge|vertex'
        """
        q = np.asarray(q_xyz, dtype=float)
        p = q[:2]
        # 2.1 Intento dentro de triángulo
        fi, bary = self.find_triangle_for_point(p)
        if fi is not None:
            z = np.dot(bary, self.z[self.F[fi]])
            foot = np.array([p[0], p[1], z], dtype=float)
            return {"foot": foot, "vertical_distance": float(abs(q[2] - z)), "where": "inside", "face": fi}

        # 2.2 Si está fuera: buscar vértice más cercano en XY
        d, vi = self._kdtree_xy.query(p)
        best = {"type": "vertex", "idx": vi, "xy": self.xy[vi], "dist2": float(d * d)}

        # 2.3 Probar aristas del 1-ring de ese vértice
        for fi in self.vert_to_faces[vi]:
            face = self.F[fi]
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                A = self.xy[a]; B = self.xy[b]
                AB = B - A; AB2 = np.dot(AB, AB)
                if AB2 < 1e-18:
                    continue
                t = np.clip(np.dot(p - A, AB) / AB2, 0.0, 1.0)
                proj = A + t * AB
                dist2 = np.dot(p - proj, p - proj)
                if dist2 < best["dist2"]:
                    best = {"type": "edge", "idx": (a, b), "xy": proj, "t": t, "dist2": float(dist2)}

        if best["type"] == "vertex":
            vi = best["idx"]
            foot = np.array([self.xy[vi, 0], self.xy[vi, 1], self.z[vi]], dtype=float)
            where = "vertex"
        else:
            a, b = best["idx"]; t = best["t"]
            xy = best["xy"]
            z = (1.0 - t) * self.z[a] + t * self.z[b]  # z lineal sobre la arista
            foot = np.array([xy[0], xy[1], z], dtype=float)
            where = "edge"

        return {"foot": foot, "vertical_distance": float(abs(q[2] - foot[2])), "where": where}

    # ------------------- (3) Shortest path (hiking) -------------------
    def _edge_cost(self, u, v):
        duv = self.V[u] - self.V[v]
        return float(np.linalg.norm(duv))

    def shortest_path(self, start_xy, goal_xy, slope_deg_max=None):
        """
        Grafo = vértices de la malla; aristas = edges de triángulos; coste = longitud 3D.
        Si slope_deg_max se define, elimina aristas cuya media de pendientes de caras adyacentes supere el umbral.
        Devuelve: path_indices, path_coords (Nx3), total_length
        """
        # Construcción de vecindad por aristas (una vez por llamada; podría cachearse)
        neighbors = defaultdict(list)
        for (u, v), faces in self.edge_to_faces.items():
            # Pruning por pendiente media
            if slope_deg_max is not None and len(faces) > 0:
                avg_slope = np.mean([self.get_slope(fi) for fi in faces])
                if avg_slope > slope_deg_max:
                    continue
            w = self._edge_cost(u, v)
            neighbors[u].append((v, w))
            neighbors[v].append((u, w))

        # Anclaje de start/goal a los vértices más cercanos en XY
        _, s_idx = self._kdtree_xy.query(np.asarray(start_xy))
        _, t_idx = self._kdtree_xy.query(np.asarray(goal_xy))

        # Dijkstra
        INF = 1e300
        dist = defaultdict(lambda: INF)
        prev = {}
        dist[s_idx] = 0.0
        pq = [(0.0, s_idx)]
        visited = set()

        while pq:
            d, u = heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            if u == t_idx:
                break
            for v, w in neighbors.get(u, []):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heappush(pq, (nd, v))

        if t_idx not in prev and t_idx != s_idx:
            return [], np.zeros((0, 3)), float("inf")

        # Reconstrucción de camino
        path_idx = [t_idx]
        while path_idx[-1] != s_idx:
            if path_idx[-1] not in prev:
                break
            path_idx.append(prev[path_idx[-1]])
        path_idx.reverse()
        path_xyz = self.V[np.array(path_idx)]

        total_len = 0.0
        for a, b in zip(path_idx[:-1], path_idx[1:]):
            total_len += self._edge_cost(a, b)
        return path_idx, path_xyz, float(total_len)

    # ------------------- (4) Contours (isohipsas) -------------------
    def contours(self, levels, tol=1e-9):
        """
        Para cada nivel c, intersecta cada triángulo con el plano z=c y ensambla segmentos en polilíneas.
        Return: dict { level : [polyline_ndarray (Kx3), ...] }
        """
        levels = [float(c) for c in levels]
        V = self.V; F = self.F
        out = {}

        for c in levels:
            segments = []  # lista de pares (p0, p1), cada uno 3D con z=c
            for a, b, d in F:
                ia, ib, id = int(a), int(b), int(d)
                za, zb, zd = V[ia, 2], V[ib, 2], V[id, 2]
                zs = np.array([za, zb, zd])
                zmin, zmax = zs.min(), zs.max()
                if c < zmin - tol or c > zmax + tol:
                    continue

                idxs = [ia, ib, id]
                # encontrar hasta dos intersecciones arista-nivel
                pts = []
                for (i, j) in ((ia, ib), (ib, id), (id, ia)):
                    zi, zj = V[i, 2], V[j, 2]
                    if (zi - c) * (zj - c) > 0:  # mismo lado -> no cruza
                        continue
                    if abs(zi - zj) < tol:
                        # Arista casi horizontal al nivel: ignora (caso degenerado poco frecuente)
                        continue
                    t = (c - zi) / (zj - zi)
                    if t < -1e-9 or t > 1 + 1e-9:
                        continue
                    P = V[i] + t * (V[j] - V[i])
                    P[2] = c
                    pts.append(P)
                    if len(pts) == 2:
                        break
                if len(pts) == 2:
                    segments.append((pts[0], pts[1]))

            # Ensamblar segmentos → polilíneas (unión por extremos con tolerancia)
            # Usamos “hash” por redondeo de XY
            def keyXY(P):
                return (round(P[0], 6), round(P[1], 6))

            adj = defaultdict(list)  # endpoint -> lista de (seg_id, endpoint_index)
            for si, (P0, P1) in enumerate(segments):
                adj[keyXY(P0)].append((si, 0))
                adj[keyXY(P1)].append((si, 1))

            used = [False] * len(segments)
            polylines = []

            for si in range(len(segments)):
                if used[si]:
                    continue
                # crecer por ambos extremos
                P0, P1 = segments[si]
                used[si] = True
                left = deque([P0]); right = deque([P1])

                # expandir desde el extremo izquierdo
                cur_key = keyXY(P0)
                while True:
                    nxt = None
                    for (sj, endj) in adj[cur_key]:
                        if used[sj]:
                            continue
                        nxt = (sj, endj)
                        break
                    if nxt is None:
                        break
                    sj, endj = nxt
                    Q0, Q1 = segments[sj]
                    used[sj] = True
                    if endj == 0:
                        left.appendleft(Q1)
                        cur_key = keyXY(Q1)
                    else:
                        left.appendleft(Q0)
                        cur_key = keyXY(Q0)

                # expandir desde el extremo derecho
                cur_key = keyXY(P1)
                while True:
                    nxt = None
                    for (sj, endj) in adj[cur_key]:
                        if used[sj]:
                            continue
                        nxt = (sj, endj)
                        break
                    if nxt is None:
                        break
                    sj, endj = nxt
                    Q0, Q1 = segments[sj]
                    used[sj] = True
                    if endj == 0:
                        right.append(Q1)
                        cur_key = keyXY(Q1)
                    else:
                        right.append(Q0)
                        cur_key = keyXY(Q0)

                poly = np.array(list(left) + list(right))
                polylines.append(poly)

            out[c] = polylines
        return out
