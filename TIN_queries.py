import numpy as np

class TINQuerier:
    def __init__(self, V, F):
        if V.shape[1] != 3 or F.shape[1] != 3:
            raise ValueError("Vértices (V) y Caras (F) deben tener 3 columnas.")
        
        self.V = V
        self.F = F
        
        # --- Optimización: Pre-calcular los vectores normales para cada cara ---
        # Esto es muy eficiente porque solo lo calculas una vez.
        # v0, v1, v2 son los tres vértices de cada triángulo
        v0 = self.V[self.F[:, 0], :]
        v1 = self.V[self.F[:, 1], :]
        v2 = self.V[self.F[:, 2], :]
        
        # Calculamos el vector normal para todas las caras a la vez usando numpy
        self.normals = np.cross(v1 - v0, v2 - v0)
        print(f"Pre-calculados {len(self.normals)} vectores normales.")

    def find_triangle_for_point(self, p_xy):

        p = np.array(p_xy, dtype=float)

        # Recorremos cada cara del TIN
        for i, face in enumerate(self.F):
            # Obtenemos los 3 vertices de la cara, pero solo sus coords XY
            v0 = self.V[face[0], :2]
            v1 = self.V[face[1], :2]
            v2 = self.V[face[2], :2]

            # Calculamos las coordenadas baricentricas (u, v, w) resolviendo un sistema lineal
            # P = w*v0 + u*v1 + v*v2  (la formulacion varia, esta es una comun)
            # Aquí usamos la fórmula directa derivada de la solución del sistema
            v10 = v1 - v0
            v20 = v2 - v0
            vp0 = p - v0

            d00 = np.dot(v10, v10)
            d01 = np.dot(v10, v20)
            d11 = np.dot(v20, v20)
            d20 = np.dot(vp0, v10)
            d21 = np.dot(vp0, v20)
            
            denom = d00 * d11 - d01 * d01
            if np.abs(denom) < 1e-10: # Triangulo degenerado
                continue

            u = (d11 * d20 - d01 * d21) / denom
            v = (d00 * d21 - d01 * d20) / denom
            w = 1.0 - u - v

            # Test de Punto-en-Triangulo: si u, v, w estan entre 0 y 1, el punto esta dentro
            if u >= -1e-6 and v >= -1e-6 and w >= -1e-6:
                return i, np.array([w, u, v])

        return None, None

    def get_slope(self, face_index):

        normal = self.normals[face_index]
        nx, ny, nz = normal[0], normal[1], normal[2]

        # Evitar division por cero si la cara es vertical
        if np.abs(nz) < 1e-9:
            return 90.0

        # Formula: slope = arctan(sqrt(nx^2 + ny^2) / |nz|)
        slope_rad = np.arctan(np.sqrt(nx**2 + ny**2) / np.abs(nz))
        return np.degrees(slope_rad)

    def get_aspect(self, face_index):
        normal = self.normals[face_index]
        nx, ny, _ = normal[0], normal[1], normal[2]
        
        # Formula: aspect = atan2(ny, nx)
        aspect_rad = np.arctan2(ny, nx)
        aspect_deg = np.degrees(aspect_rad)

        # Convertir de [-180, 180] a [0, 360]
        if aspect_deg < 0:
            aspect_deg += 360
            
        return aspect_deg
        
    def interpolate_height(self, p_xy):
        face_index, bary_coords = self.find_triangle_for_point(p_xy)

        if face_index is None:
            return None # El punto esta fuera de la malla

        # Obtener los 3 vertices completos (XYZ) de la cara
        face_vertices_indices = self.F[face_index]
        face_vertices = self.V[face_vertices_indices]

        # Ponderar las coordenadas Z de los vértices con las coordenadas baricentricas
        # z_interp = w*z0 + u*z1 + v*z2
        z_interp = np.dot(bary_coords, face_vertices[:, 2])
        
        return z_interp
    