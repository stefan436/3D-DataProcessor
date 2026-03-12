# src/core/meshing.py

import os
import sys
import gc
import math
import struct
import shutil
import multiprocessing

import numpy as np
import open3d as o3d
import trimesh
import pymeshlab
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from tqdm import tqdm


def _generate_grid_faces_from_mask(valid_mask):
    """
    Erzeugt Faces nur für valide Pixel basierend auf einer Boolean-Maske.
    Nutzt eine Index-Lookup-Tabelle (Mapping), um 2D-Grid-Koordinaten
    auf komprimierte 1D-Vertex-Indizes abzubilden.
    
    Args:
        valid_mask (np.ndarray): 2D Boolean Array (True = Vertex existiert).
        
    Returns:
        np.ndarray: (N, 3) Array mit Face-Indizes.
    """
    rows, cols = valid_mask.shape
    
    # 1. Index-Mapping erstellen (Lookup Table)
    # Wir erstellen ein Grid gleicher Größe, initialisiert mit -1 (ungültig).
    # Nur Positionen, wo valid_mask True ist, bekommen eine fortlaufende ID.
    id_map = np.full((rows, cols), -1, dtype=np.int32)
    
    # Anzahl der validen Vertices bestimmen (das wird die Länge des Vertex-Arrays)
    num_valid = np.count_nonzero(valid_mask)
    
    # Die validen Positionen durchnummerieren (0 bis N-1)
    id_map[valid_mask] = np.arange(num_valid, dtype=np.int32)
    
    # 2. Vektorisierte Nachbarn holen (Slicing)
    # TL = Top-Left, BR = Bottom-Right, etc.
    # Wir betrachten immer das Quadrat zwischen (r, c) und (r+1, c+1)
    tl = id_map[:-1, :-1].ravel()
    tr = id_map[:-1, 1:].ravel()
    bl = id_map[1:, :-1].ravel()
    br = id_map[1:, 1:].ravel()
    
    # 3. Faces konstruieren
    # Ein Quad besteht aus zwei Dreiecken. Ein Dreieck ist nur valide, 
    # wenn ALLE 3 Ecken im id_map einen Wert >= 0 haben.
    
    # Triangle 1: TL -> BL -> TR
    # Maske: Alle drei Ecken müssen valide sein
    mask_t1 = (tl >= 0) & (bl >= 0) & (tr >= 0)
    # Indizes stapeln
    f1 = np.column_stack((tl[mask_t1], bl[mask_t1], tr[mask_t1]))
    
    # Triangle 2: TR -> BL -> BR
    mask_t2 = (tr >= 0) & (bl >= 0) & (br >= 0)
    f2 = np.column_stack((tr[mask_t2], bl[mask_t2], br[mask_t2]))
    
    # 4. Zusammenfügen
    return np.vstack((f1, f2))

def _convert_open3d_to_trimesh(o3d_mesh):
    """Konvertiert Open3D Mesh zu Trimesh."""
    if not o3d_mesh.has_vertices() or not o3d_mesh.has_triangles(): return None
    v = np.asarray(o3d_mesh.vertices)
    f = np.asarray(o3d_mesh.triangles)
    vc = np.asarray(o3d_mesh.vertex_colors)
    if vc.shape[0] > 0: vc = (vc * 255).astype(np.uint8)
    else: vc = None
    return trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vc, process=False, validate=False)

def _remove_invalid_and_duplicate_points(pcd):
    """Bereinigt Punktwolke (nur NaNs und Duplikate, KEIN Downsampling)."""
    # Entfernt Punkte mit Koordinaten wie NaN oder Inf
    pcd = pcd.remove_non_finite_points()
    
    # Entfernt exakt identische Punkte (doppelte Koordinaten)
    pcd = pcd.remove_duplicated_points()
    return pcd

def _estimate_and_orient_normals(pcd, scan_type='aerial'):
    """Berechnet Normalen (MST orientiert) falls nicht vorhanden."""
    if pcd.has_normals():
        print("  [Normals] Verwende existierende Normalen aus Import.", flush=True)
        return pcd

    print("  [Normals] Berechne neu (CloudCompare Strategy)...", flush=True)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances) if len(distances) > 0 else 0.05
    search_radius = avg_dist * 5.0

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=search_radius, max_nn=30
        )
    )

    print("  [Normals] Orientiere via MST (Consistent Tangent Plane)...", flush=True)
    pcd.orient_normals_consistent_tangent_plane(k=30)

    if scan_type == 'aerial':
        normals = np.asarray(pcd.normals)
        if np.mean(normals[:, 2]) < 0:
            print("  [Normals] Invertiere Ausrichtung (Aerial Correction)", flush=True)
            pcd.normals = o3d.utility.Vector3dVector(-normals)

    return pcd

def _generate_mesh_from_npz(npz_path, algorithm='poisson', depth=10, pointweight=6, samplespernode=1.0, scan_type='auto',
                           hc_laplacian_smoothing=False, decimation=False, percentile_of_faces=0.9, 
                           repair_non_manifold=True):
    """Kernfunktion für das Meshing einer einzelnen NPZ-Datei mit Post-Processing Pipeline.

    Verarbeitet Punktwolken zu Meshes und wendet eine Kette von geometrischen Korrekturen an.
    
    Ablauf der Pipeline:
    1. Koordinaten-Shift: Verschiebt Daten zum Ursprung (0,0,0), um Float32-Präzisionsfehler zu vermeiden.
    2. Meshing:
       - Grid: Direkte Triangulierung valider Pixel (Masked Array).
       - Raw: Poisson/BPA Rekonstruktion + Artifact-Cropping (Bounding Box + 5% Margin) + Optionales Smoothing.
    3. Optional: Smoothing (HC Laplacian) zur Rauschunterdrückung unter Beibehaltung des Volumens.
    4. Optional: Decimation (Quadric Edge Collapse) zur Reduktion von "Treppeneffekten" auf planaren Flächen.
    5. Reverse-Shift: Schiebt das fertige Mesh zurück an die ursprüngliche Geoposition.

    Args:
        npz_path (str): Pfad zur Quelldatei.
        algorithm (str): 'poisson' (Robust, wasserdicht) oder 'bpa' (Detailtreu, kann Löcher haben).
        depth (int): Octree Tiefe (Auflösung) für Poisson. Erhöht RAM-Bedarf exponentiell (2^d).
        pointweight (float): Gewichtung der Originalpunkte gegenüber der Glättung.
        samplespernode (float): Rauschfilterung für Poisson (höher = weniger Rauschen, weniger Details).
        scan_type (str): 'auto', 'aerial', 'terrestrial'. Steuert Normalen-Ausrichtung.
        hc_laplacian_smoothing (bool): Aktiviert volumen-erhaltendes Glätten (gut für verrauschte Wände).
        decimation (bool): Aktiviert Mesh-Vereinfachung (gut gegen Aliasing/Treppenstufen).
        percentile_of_faces (float): Ziel-Komplexität bei Decimation (0.9 = behalte 90% der Faces).

    Returns:
        trimesh.Trimesh: Das prozessierte Mesh Objekt oder None bei Fehler/leeren Daten.
    """
    print(f"[{os.path.basename(npz_path)}] Starte Verarbeitung...", flush=True)
    
    mesh = None 
    centroid = None

    try:
        loaded = np.load(npz_path, allow_pickle=True)
        data_type = str(loaded['type'])

        # ==================================================================================
        # PFAD A: GRID (RASTER)
        # ==================================================================================
        if data_type == 'grid':
            print(f"[{os.path.basename(npz_path)}] Modus: GRID (Optimized Masking)", flush=True)
            z = loaded['data']; meta = loaded['meta']
            colors = loaded['colors'] if 'colors' in loaded else None
            
            # Maske basierend auf NaN im Z-Grid
            valid_mask = ~np.isnan(z)
            if not np.any(valid_mask): return None

            # 1. Koordinaten & Topologie (Vektorisieren)
            rows, cols = z.shape
            x_lin = meta[0] + np.arange(cols) * meta[2] + (meta[2]/2)
            y_lin = meta[1] + np.arange(rows) * meta[3] + (meta[3]/2)
            xg, yg = np.meshgrid(x_lin, y_lin)
            
            verts = np.column_stack((xg[valid_mask], yg[valid_mask], z[valid_mask])).astype(np.float32)
            faces = _generate_grid_faces_from_mask(valid_mask)
            
            if len(faces) == 0: return None

            # 2. Shift
            centroid = np.mean(verts, axis=0)
            verts -= centroid
            
            # Farben vorbereiten (Trimesh Format: uint8)
            v_colors = None
            if colors is not None:
                c_valid = colors[valid_mask]
                if c_valid.max() <= 1.0: c_valid *= 255
                v_colors = c_valid.astype(np.uint8)

            # --- BRANCH: DECIMATION ---
            if decimation:
                print(f"  [Decimation] Reduziere Grid auf {percentile_of_faces*100}%...", flush=True)
                try:
                    ms = pymeshlab.MeshSet()
                    
                    # FIX: Dynamische Argumente für den Konstruktor
                    mesh_kwargs = {
                        'vertex_matrix': verts,
                        'face_matrix': faces.astype(np.int32)
                    }
                    
                    # Farben NUR hinzufügen, wenn sie existieren (Kein None übergeben!)
                    if v_colors is not None:
                        c_float = v_colors.astype(np.float64) / 255.0
                        # Alpha Kanal hinzufügen für PyMeshLab (RGBA)
                        pm_colors = np.hstack((c_float, np.ones((len(c_float), 1))))
                        mesh_kwargs['v_color_matrix'] = pm_colors

                    # Mesh mit **kwargs erstellen
                    m = pymeshlab.Mesh(**mesh_kwargs)
                    ms.add_mesh(m)
                    
                    # Decimation Filter
                    target_faces = int(ms.current_mesh().face_number() * percentile_of_faces)
                    ms.meshing_decimation_quadric_edge_collapse(
                        targetfacenum=target_faces, 
                        preserveboundary=True, 
                        preservenormal=True, 
                        planarquadric=True
                    )
                    
                    # --- NEUER REPARATUR-BLOCK ---
                    if repair_non_manifold:
                        print("  [Repair] Repariere Non-Manifold Geometrien...", flush=True)
                        try:
                            ms.meshing_repair_non_manifold_edges()
                            ms.meshing_repair_non_manifold_vertices()
                        except Exception as e:
                            print(f"  [Warnung] Non-Manifold Repair Failed: {e}", flush=True)
                            
                    # Ergebnis zurückholen
                    out = ms.current_mesh()
                    verts = out.vertex_matrix() # Float64 zentriert
                    faces = out.face_matrix()
                    
                    if out.has_vertex_color():
                        c_out = out.vertex_color_matrix()
                        if c_out.max() <= 1.0: c_out *= 255
                        v_colors = c_out.astype(np.uint8)
                        
                except Exception as e:
                    print(f"  [Warnung] Grid Decimation Failed: {e}", flush=True)
                    # Fallback: Wir nutzen die originalen verts/faces weiter

            # 3. Finales Trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_colors, process=False)

        # ==================================================================================
        # PFAD B: RAW (PUNKTWOLKE)
        # ==================================================================================
        elif data_type == 'raw':
            print(f"[{os.path.basename(npz_path)}] Modus: RAW (3D Poisson)", flush=True)
            pts = loaded['data']
            if len(pts) < 10: return None

            # --- Shift -> Cast to Float32 ---
            centroid = np.mean(pts, axis=0)
            pts_centered = (pts - centroid).astype(np.float32)
            del pts # RAM freigeben

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_centered)
            
            if 'colors' in loaded:
                c = loaded['colors']
                if c.max() > 1.0: c /= 255.0
                pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float32))
            if 'normals' in loaded:
                pcd.normals = o3d.utility.Vector3dVector(loaded['normals'].astype(np.float32))
                
            pcd = _remove_invalid_and_duplicate_points(pcd)
            if not pcd.has_points():
                print(f"[{os.path.basename(npz_path)}] Leer nach Sanitization!", flush=True)
                return None

            if scan_type == 'auto':
                extent = pts_centered.max(axis=0) - pts_centered.min(axis=0)
                if max(extent[0], extent[1]) > extent[2] * 5.0:
                    scan_type = 'aerial'
                    print("  [Auto] Erkannt: AERIAL (Landschaft)", flush=True)
                else:
                    scan_type = 'terrestrial'
                    print("  [Auto] Erkannt: TERRESTRIAL (Objekt/Scan)", flush=True)

            pcd = _estimate_and_orient_normals(pcd, scan_type=scan_type)

            if algorithm == 'bpa':
                # BPA machen wir weiterhin in Open3D (schneller für BPA)
                distances = pcd.compute_nearest_neighbor_distance()
                if len(distances) > 0:
                    avg_dist = np.mean(distances)
                    radii = [avg_dist, avg_dist * 2, avg_dist * 4]
                    o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii))
                    mesh = _convert_open3d_to_trimesh(o3d_mesh)

            else: # POISSON via PyMeshLab
                try:
                    # Daten aus Open3D extrahieren
                    v_arr = np.asarray(pcd.points).astype(np.float64)
                    n_arr = np.asarray(pcd.normals).astype(np.float64)
                    
                    # Dictionary für Konstruktor-Argumente erstellen
                    # KORREKTUR 1: 'v_normals_matrix' statt 'normal_matrix'
                    mesh_kwargs = {
                        'vertex_matrix': v_arr,
                        'v_normals_matrix': n_arr 
                    }

                    # KORREKTUR 2: Farben nur hinzufügen, wenn vorhanden (kein None übergeben)
                    if pcd.has_colors():
                        c_rgb = np.asarray(pcd.colors)
                        # Alpha hinzufügen (1.0)
                        c_rgba = np.hstack((c_rgb, np.ones((c_rgb.shape[0], 1))))
                        # PyMeshLab erwartet float64 RGBA
                        mesh_kwargs['v_color_matrix'] = c_rgba.astype(np.float64)

                    # MeshSet erstellen
                    ms = pymeshlab.MeshSet()
                    
                    # Mesh mit **kwargs initialisieren
                    m = pymeshlab.Mesh(**mesh_kwargs)
                    ms.add_mesh(m)

                    # Poisson Filter ausführen
                    print(f"  [PyMeshLab] Poisson Recon (Depth={depth}, Point weight={pointweight}, Samples per Node={samplespernode})...", flush=True)
                    ms.generate_surface_reconstruction_screened_poisson(
                        depth=depth, 
                        pointweight=pointweight, 
                        samplespernode=samplespernode, 
                        preclean=True
                    )
                    
                    # --- Geometrisches Cropping (Bounding Box + Margin) ---
                    print("  [PyMeshLab] Bounding Box Crop...", flush=True)

                    # 1. Grenzen der ORIGINAL-Punkte ermitteln (im zentrierten Raum)
                    min_b = pcd.get_min_bound()
                    max_b = pcd.get_max_bound()

                    # 2. Sicherheitsabstand (Margin) berechnen (z.B. 5% der Größe)
                    # WICHTIG: Ohne Margin schneidest du verrauschte Wände kaputt!
                    extent = max_b - min_b
                    margin_ratio = 0.05  # 5% Puffer
                    margin = extent * margin_ratio

                    limit_min = min_b - margin
                    limit_max = max_b + margin

                    # 3. Bedingung für PyMeshLab erstellen
                    # Logik: WENN (x < min) ODER (x > max) ODER (y < min) ... DANN selektieren
                    cond_str = (
                        f"(x < {limit_min[0]}) || (x > {limit_max[0]}) || "
                        f"(y < {limit_min[1]}) || (y > {limit_max[1]}) || "
                        f"(z < {limit_min[2]}) || (z > {limit_max[2]})"
                    )

                    # 4. Selektieren und Löschen
                    ms.compute_selection_by_condition_per_vertex(condselect=cond_str)
                    ms.meshing_remove_selected_vertices()

                    # --- HC Laplacian Smoothing ---
                    if hc_laplacian_smoothing:
                        print("  [PyMeshLab] HC Laplacian Smoothing...", flush=True)
                        
                        try:
                            ms.apply_coord_hc_laplacian_smoothing()
                        except Exception as e_smooth:
                            print(f"  [Warnung] Smoothing fehlerhaft: {e_smooth}")
                            
                        
                    if decimation:
                        print("  [PyMeshLab] Quadric Edge Collapse Decimation...", flush=True)
                        
                        # Wir reduzieren die Anzahl der Faces. Das entfernt Rauschen auf geraden Flächen
                        # sehr effektiv, behält aber scharfe Hauskanten bei.
                        
                        current_faces = ms.current_mesh().face_number()
                        target_faces = int(current_faces * percentile_of_faces) # Reduzierung auf 50% (aggressiv: 0.2)

                        try:
                            ms.meshing_decimation_quadric_edge_collapse(
                                targetfacenum=target_faces, 
                                preserveboundary=True,    # WICHTIG: Ränder des Crops nicht zerfressen
                                preservenormal=True,      # Hilft, die Orientierung der Wände zu behalten
                                planarquadric=True,        # Optimiert speziell für flache Wände (Architektur!)
                                autoclean=True
                            )
                            
                            # --- NEUER REPARATUR-BLOCK ---
                            if repair_non_manifold:
                                print("  [Repair] Repariere Non-Manifold Geometrien...", flush=True)
                                try:
                                    ms.meshing_repair_non_manifold_edges()
                                    ms.meshing_repair_non_manifold_vertices()
                                except Exception as e:
                                    print(f"  [Warnung] Non-Manifold Repair Failed: {e}", flush=True)
                                    
                        except Exception as e_dec:
                            print(f"  [Warnung] Decimation fehlgeschlagen: {e_dec}")


                    # Ergebnis holen
                    out_m = ms.current_mesh()
                    
                    # Konvertierung zu Trimesh
                    verts = out_m.vertex_matrix()
                    faces = out_m.face_matrix()
                    
                    # Farben zurückholen
                    v_colors_out = None
                    if out_m.has_vertex_color():
                        c_out = out_m.vertex_color_matrix()
                        # Float [0,1] -> Byte [0,255]
                        if c_out.max() <= 1.0: c_out *= 255
                        v_colors_out = c_out.astype(np.uint8)

                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=v_colors_out, process=False)
                    
                except Exception as e_poi:
                    print(f"PYMESHLAB CRASH: {e_poi}")
                    return None

        # --- 3. Shift zurücksetzen ---
        if mesh is not None:
            mesh.vertices += centroid
            print(f"[{os.path.basename(npz_path)}] Fertig ({len(mesh.vertices)} verts).", flush=True)
            return mesh

        return None


    except Exception as e:
        print(f"FATAL Error in {os.path.basename(npz_path)}: {e}", flush=True)
        return None

def export_mesh_from_npz_to_ply(npz_path, output_ply_path, algorithm='poisson', scan_type='auto', depth=10,
                         pointweight=6, samplespernode=1.0, hc_laplacian_smoothing=False, decimation=False,
                         percentile_of_faces=0.9, repair_non_manifold=True):
    """Worker-Funktion für Multiprocessing: Erstellt Mesh und speichert als PLY."""
    gc.collect(); sys.stdout.flush()
    mesh = _generate_mesh_from_npz(npz_path, algorithm=algorithm, depth=depth, pointweight=pointweight,
                                  samplespernode=samplespernode, scan_type=scan_type, hc_laplacian_smoothing=hc_laplacian_smoothing,
                                  decimation=decimation, percentile_of_faces=percentile_of_faces, repair_non_manifold=repair_non_manifold)
    if mesh is not None and len(mesh.vertices) > 0:
        mesh.export(output_ply_path)
        return True
    return False

def _merge_and_snap_mesh_pair(file_a, file_b, out_path, gap_threshold=1.5, perform_snapping=True, close_artifacts=False):
    """Hilfsfunktion: Lädt zwei Meshes, fügt sie zusammen und speichert das Ergebnis.
    
    Diese Funktion ist die "Atomare Einheit" des Merge-Trees.
    Sie lädt nur das Nötigste in den RAM, verarbeitet es und löscht es sofort wieder.
    """
    try:
        m1 = trimesh.load(file_a, process=False)
        m2 = trimesh.load(file_b, process=False)
        
        # Performance-Check: Snapping bei riesigen Dateien deaktivieren?
        # Hier lassen wir es an, da der User Qualität will, aber wir warnen bei RAM-Spikes.
        
        # Snapping Logic (Portiert aus V7.8)
        if perform_snapping and not m1.is_empty and not m2.is_empty:
            bounds = m2.bounds
            bounds_min = bounds[0] - gap_threshold
            bounds_max = bounds[1] + gap_threshold
            
            # Nur Vertices von M1 prüfen, die nahe an M2 liegen
            m1_verts = m1.vertices
            mask = np.all((m1_verts >= bounds_min) & (m1_verts <= bounds_max), axis=1)
            
            if np.any(mask):
                nearby_verts = m1_verts[mask]
                # KDTree nur für den relevanten Teil bauen (viel schneller)
                tree = cKDTree(nearby_verts[:, :2]) # 2D Snapping (XY) oft ausreichend für Terrain
                dists, idxs = tree.query(m2.vertices[:, :2], distance_upper_bound=gap_threshold)
                
                valid_snaps = dists < gap_threshold
                if np.any(valid_snaps):
                    # Setze Koordinaten von M2 auf die von M1 (Snap)
                    m2.vertices[valid_snaps] = nearby_verts[idxs[valid_snaps]]

        # Concatenate
        combined = trimesh.util.concatenate([m1, m2])
        if close_artifacts:
            combined.process(validate=True)
            trimesh.repair.fill_holes(combined)

        if perform_snapping:
            combined.merge_vertices() # Entfernt Duplikate nach dem Snap
            
        combined.export(out_path)
        
        # Cleanup RAM
        del m1, m2, combined
        gc.collect()
        return True
    except Exception as e:
        print(f"Fehler beim Merge-Pair ({os.path.basename(file_a)} + {os.path.basename(file_b)}): {e}", flush=True)
        return False

def merge_meshes_hierarchically(file_list, output_file, temp_dir, gap_threshold=1.5, perform_snapping=True, close_artifacts=False):
    """Führt einen speicherschonenden Merge-Tree (Hierarchischer Merge) durch.

    Kombiniert Meshes paarweise in einem Binärbaum-Verfahren (O(N log N)).
    
    Vorteile gegenüber iterativem Merge (A + B -> res + C -> res ...):
    1. Numerische Stabilität: Koordinaten werden seltener transformiert/addiert.
    2. RAM-Effizienz: Es befinden sich immer nur zwei Teilstücke gleichzeitig im Speicher.
    3. Disk-Offloading: Zwischenergebnisse ("Generationen") werden temporär auf die Festplatte geschrieben.

    Args:
        file_list (list): Liste von Pfaden zu PLY-Dateien.
        output_file (str): Pfad zur finalen Ausgabedatei.
        temp_dir (str): Ordner für temporäre Dateien (wird nach Abschluss bereinigt).
        gap_threshold (float): Maximalabstand für Vertex-Snapping (Lückenschluss) an Kachelrändern.
        perform_snapping (bool): Wenn True, werden nahe beieinander liegende Vertices an Kanten verschmolzen (Stitching).
    """
    current_generation = file_list
    gen_idx = 0
    
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Starte Hierarchical Merge mit {len(file_list)} Dateien...", flush=True)

    while len(current_generation) > 1:
        next_generation = []
        gen_idx += 1
        num_pairs = math.ceil(len(current_generation) / 2)
        
        print(f"--- Generation {gen_idx}: Verarbeite {len(current_generation)} Dateien in {num_pairs} Paaren ---", flush=True)
        
        # Iteriere in 2er Schritten
        for i in tqdm(range(0, len(current_generation), 2), desc=f"Gen {gen_idx}"):
            file_a = current_generation[i]
            
            # Fall: Ungerade Anzahl, letztes Element wird einfach durchgereicht
            if i + 1 >= len(current_generation):
                next_generation.append(file_a)
                continue
                
            file_b = current_generation[i+1]
            
            # Temporärer Dateiname
            temp_name = f"gen_{gen_idx}_pair_{i//2}.ply"
            temp_path = os.path.join(temp_dir, temp_name)
            
            success = _merge_and_snap_mesh_pair(file_a, file_b, temp_path, gap_threshold, perform_snapping, close_artifacts)
            
            if success:
                next_generation.append(temp_path)
            else:
                # Fallback bei Fehler: Beide Originaldateien weiterreichen (besser als Datenverlust)
                print(f"Warnung: Merge fehlgeschlagen, behalte Einzeldateien.", flush=True)
                next_generation.append(file_a)
                next_generation.append(file_b)
        
        current_generation = next_generation
        gc.collect() # Wichtig!
        
    # Finales Ergebnis verschieben
    if current_generation:
        print(f"Verschiebe finales Ergebnis nach: {output_file}", flush=True)
        shutil.move(current_generation[0], output_file)
        
        # Cleanup Temp
        try:
            shutil.rmtree(temp_dir)
            print("Temp-Ordner bereinigt.", flush=True)
        except Exception as e:
            print(f"Konnte Temp-Ordner nicht löschen: {e}", flush=True)
    else:
        print("Fehler: Keine Dateien übrig geblieben.", flush=True)

def batch_process_npz_to_meshes(input_folder, output_folder, combine=True, combined_filename="combined_model.ply",
                 algorithm='poisson', scan_type='aerial', depth=10, pointweight=6, samplespernode=1.0,
                 hc_laplacian_smoothing=False, decimation=False, percentile_of_faces=0.9,
                 debug=False, gap_threshold=1.5, close_artifacts=False, repair_non_manifold=True):
    """
    Verarbeitung von npz zu meshes.
    Steuert den gesamten Prozess:
    1. Parallelisierte Umwandlung aller .npz Dateien in Einzel-Meshes (Worker-Pool).
    2. Intelligente Erkennung des Datentyps:
       - 'grid' (2.5D): Aktiviert Snapping/Stitching für nahtlose Übergänge.
       - 'raw' (3D): Deaktiviert Snapping, da Überlappungen hier komplexer sind.
    3. Hierarchisches Zusammenfügen aller Ergebnisse zu einem Gesamtmodell.

    Args:
        input_folder (str): Pfad zum Ordner, der die Quelldateien (.npz) enthält.
        output_folder (str): Zielordner, in dem die generierten .ply-Dateien gespeichert werden.
        combine (bool, optional): Fügt alle Meshes am Ende zu einer Datei zusammen (`combined_filename`).
            Defaults to True.
        combined_filename (str, optional): Dateiname für das zusammengefügte Mesh (nur relevant,
            wenn `combine=True`).
            Defaults to "combined_model.ply".
        algorithm (str, optional): Der Meshing-Algorithmus (nur relevant für RAW daten)
            - 'poisson': (Empfohlen) Screened Poisson Reconstruction. Erzeugt wasserdichte,
              glatte Oberflächen. Gut für unvollständige Daten.
            - 'bpa': Ball-Pivoting Algorithm. Verbindet Punkte direkt. Erhält Details besser,
              hinterlässt aber Löcher, wenn die Punktdichte zu gering ist.
            Defaults to 'poisson'.
        scan_type (str, optional): Art der Datenaufnahme zur Orientierung der Normalen.
            - 'aerial': Für Luftbilder/Drohnen - ALS (Normalen zeigen nach oben).
            - 'terrestrial': Für Boden-Scans - TLS.
            - 'auto': Versucht, anhand der Bounding-Box das Format zu erraten.
            Defaults to 'aerial'.
        depth (int, optional): Octree-Tiefe für den Poisson-Algorithmus. Steuert die Auflösung.
            RAM verbrauch ist exponential mit der depth.
            - 8-9: Grob, sehr schnell.
            - 10: Guter Standard für Gelände.
            - 11-12: Hohe Details, sehr hoher RAM-Verbrauch.
            Defaults to 10.
        PointWeight (float, optional): Spezifiziert die Wichtigkeit der Messpunkte in der Reconstruction.
            - 0-4: Mehr glättung
            - 4-10: Algorithmus hält sich strenger an die Messpunkte; weniger smooth.
            Defaults to 6.
        SamplesPerNode (float, optional): Spezifiziert die Mindestanzahl von Messpunkten, die innerhalb eines Octree-Knotens liegen müssen, damit er berücksichtigt wird.
            - < 1: Rauschfreie Messungen
            - 1-5: Für Messungen mit wenig Rauschen
            - 15-20: Für sehr rauschige Messungen
        hc_laplacian_smoothing (bool, optional): Aktiviert Laplace Filter. Relativ starker smoothing effekt.
            Defaults to False
        decimation (bool, optional): Aktiviert Quadric Edge Collapse Decimation. Verringert Anzahl der Faces durch Vereinachung des meshes.
            Defaults to False.
        decimation (bool, optional): Aktiviert Quadric Edge Collapse Decimation für ALLE Datentypen (Grid & Raw).
            Verringert die Anzahl der Faces durch Vereinfachung des Meshes, wobei Kachel-Ränder
            geschützt werden (Preserve Boundary), um Stitching zu ermöglichen.
            Defaults to False.
        percentile_of_faces (float, optional): Prozentuale Anzahl der originalen Faces, die nach der Vereinfachung noch vorhanden sein soll (0.9 = 90%).
            Defaults to 0.9.
        debug (bool, optional): Steuert den Ausführungsmodus.
            - True: Single-Threaded (Sequenziell).
            - False: Multi-Processing (Parallel).
            Defaults to False.
        gap_threshold (float, optional): Toleranzradius in Modelleinheiten für das Schließen von Lücken
            (Stitching) zwischen Kacheln.
            Punkte an den Rändern, die horizontal (X/Y) näher als dieser Wert beieinander liegen,
            werden zusammengezogen (Vertex-Snapping), um ein wasserdichtes Modell zu garantieren.
            Defaults to 1.5.
        close_artifacts (bool, optional): Ob nach Erstellung jedes Meshes bei dem hierarchical_merge eine Reparatur mit close holes laufen soll.
            Relevant wenn combine=True.
            Defaults to False.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Start Meshing (Type={scan_type}, Depth={depth}), Point weight={pointweight}, Samples per Node={samplespernode}\n", flush=True)

    files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz')]
    num_files = len(files)

    if num_files == 0:
        print("Keine .npz Dateien im Eingabeordner gefunden.", flush=True)
        return

    # --- Pre-Check: Datentyp erkennen ---
    is_grid_data = False
    try:
        with np.load(files[0], allow_pickle=True) as d:
            if str(d['type']) == 'grid':
                is_grid_data = True
                print("[Info] Grid-Daten erkannt.", flush=True)
                print(f"Starte Meshing", flush=True)
            else:
                print("[Info] Raw-Daten erkannt.", flush=True)
                print(f"Starte Meshing (Algo={algorithm}, Type={scan_type}, Depth={depth}, Point weight={pointweight}, Samples per Node={samplespernode})", flush=True)
    except Exception as e:
        print(f"[Warnung] Konnte Datentyp nicht bestimmen ({e}). Deaktiviere Snapping sicherheitshalber.", flush=True)
        print(f"Versuche Meshing", flush=True)

    generated_meshes = []
    args = []

    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out_name = os.path.join(output_folder, f"{base}_mesh.ply")
        args.append((f, out_name, algorithm, scan_type, depth, pointweight, samplespernode, hc_laplacian_smoothing, decimation, percentile_of_faces, repair_non_manifold))
        generated_meshes.append(out_name)

    # --- Phase 1: Individuelle Meshes erstellen ---
    if debug:
        print("DEBUG MODE: Running Single-Threaded loop...", flush=True)
        for arg in args:
            export_mesh_from_npz_to_ply(*arg)
    else:
        cpu_cores = multiprocessing.cpu_count()
        num_processes = max(1, min(num_files, cpu_cores))
        
        print(f"Starten von {num_processes} Worker-Prozessen (für {num_files} Dateien)...", flush=True)

        with multiprocessing.Pool(processes=num_processes) as pool:
            list(tqdm(pool.starmap(export_mesh_from_npz_to_ply, args), total=len(args), desc="Meshing (Multi)"))

    # --- Phase 2: Kombinieren (mit Merge-Tree) ---
    if combine:
        print("\nStarte Zusammenfügen (Hierarchischer Merge)...", flush=True)
        valid_meshes = [f for f in generated_meshes if os.path.exists(f)]

        if len(valid_meshes) < len(generated_meshes):
            print(f"WARNUNG: Nur {len(valid_meshes)}/{len(generated_meshes)} Meshes erstellt.", flush=True)

        if not valid_meshes:
            print("Keine Meshes zum Kombinieren vorhanden.")
            return
        
        if not os.path.splitext(combined_filename)[1]:
            combined_filename += ".ply"
        save_path = os.path.join(output_folder, combined_filename)
        
        # Temp Folder für Merge-Zwischenschritte
        temp_merge_dir = os.path.join(output_folder, "temp_merge_processing")

        merge_meshes_hierarchically(
            valid_meshes, 
            save_path, 
            temp_merge_dir, 
            gap_threshold=gap_threshold, 
            perform_snapping=is_grid_data,
            close_artifacts=close_artifacts
        )

def concatenate_ply_files_memory_efficiently(file1, file2, output_path):
    """Kombiniert zwei PLY-Dateien RAM-effizient zu einer Datei.

    Lädt zwei Meshes, vereint Vertices und Faces und speichert das Ergebnis.
    Löscht Quellobjekte sofort aus dem RAM.
    Warnung: Vertex-Farben werden aktuell nicht unterstützt.

    Args:
        file1 (str): Pfad zur ersten PLY Datei.
        file2 (str): Pfad zur zweiten PLY Datei.
        output_path (str): Zielpfad der kombinierten Datei.
    """
    import gc

    print(f"Lade Datei 1: {os.path.basename(file1)}", flush=True)
    m1 = trimesh.load(file1, process=False)
    v1 = m1.vertices
    f1 = m1.faces
    len_v1 = len(v1)

    print(f"Lade Datei 2: {os.path.basename(file2)}", flush=True)
    m2 = trimesh.load(file2, process=False)
    v2 = m2.vertices
    f2 = m2.faces

    # Speicheroptimierung: Indizes verschieben
    # Wenn weniger als 4 Mrd Vertices, reicht uint32 (spart 50% RAM bei Faces)
    print("Optimiere und Merge...", flush=True)
    total_verts = len(v1) + len(v2)
    dtype_faces = np.int64
    if total_verts < 2**31 - 1:
         dtype_faces = np.int32

    f2 = f2 + len_v1

    # Stacken
    v_all = np.vstack((v1, v2))
    f_all = np.vstack((f1, f2)).astype(dtype_faces)

    # WICHTIG: Alte Objekte löschen!
    del m1, m2, v1, v2, f1, f2
    gc.collect()

    # Export
    print(f"Exportiere nach {output_path}...", flush=True)
    m_final = trimesh.Trimesh(vertices=v_all, faces=f_all, process=False)
    m_final.export(output_path)
    print("Fertig.", flush=True)

def extract_boundary_edges_memory_efficiently(mesh):
    """
    Findet offene Kanten (Boundaries) mit minimalem RAM-Verbrauch.
    Ersetzt trimesh.edges_unique (welches int64 nutzt und viel RAM frisst).
    
    Strategie:
    1. Downcast Faces zu uint32.
    2. Sortiere Kanten (u < v).
    3. Packe Kanten in 1D int64 ((u << 32) | v).
    4. np.unique auf 1D Array (massiv schneller & speicherschonender).
    """
    # 1. Zugriff auf Raw Faces und sofortiger Downcast auf uint32
    # trimesh lädt faces oft als int64, was wir hier nicht brauchen.
    faces = mesh.faces.astype(np.uint32)

    # 2. Kanten konstruieren
    # Ein Face (0, 1, 2) hat die Kanten (0,1), (1,2), (2,0).
    # Wir bauen drei Vektoren für u und v.
    c0 = faces[:, 0]
    c1 = faces[:, 1]
    c2 = faces[:, 2]

    # Clean up faces array immediately
    del faces

    # u und v Vektoren erstellen (Stacking via Concatenate ist RAM-freundlicher als 2D Reshape hier)
    u = np.concatenate([c0, c1, c2])
    v = np.concatenate([c1, c2, c0])

    del c0, c1, c2

    # 3. Kanten sortieren (u < v) für Eindeutigkeit
    # Wir nutzen Bit-Masken statt np.sort(axis=1), um temporäre 2D-Arrays zu vermeiden.
    swap_mask = u > v
    
    # Casting auf uint64 ist nötig für den Bit-Shift im nächsten Schritt
    mins = np.where(swap_mask, v, u).astype(np.uint64)
    maxs = np.where(swap_mask, u, v).astype(np.uint64)

    del u, v, swap_mask

    # 4. 1D Bit-Packing
    # Kombiniert zwei 32-bit ints in einen 64-bit int.
    # Das reduziert das Problem von "Finde Duplikate in (N,2) Array" auf "Finde Duplikate in (N,) Array".
    packed_edges = (mins << 32) | maxs
    
    del mins, maxs

    # 5. Unique & Counts
    # Das ist die speicherintensivste Operation, aber dank 1D und uint64 optimiert.
    unique_packed, counts = np.unique(packed_edges, return_counts=True)

    del packed_edges

    # 6. Filtern: Boundaries sind Kanten, die nur 1x vorkommen
    boundary_packed = unique_packed[counts == 1]

    del unique_packed, counts
    
    # 7. Unpacking
    if len(boundary_packed) == 0:
        return np.empty((0, 2), dtype=np.uint32)

    u_out = (boundary_packed >> 32).astype(np.uint32)
    v_out = (boundary_packed & 0xFFFFFFFF).astype(np.uint32)
    
    del boundary_packed
    gc.collect()

    return np.column_stack((u_out, v_out))

def _sort_edges_into_closed_path(boundary_edges):
    """Sortiert unsortierte Kanten zu einem geschlossenen Pfad."""
    if len(boundary_edges) == 0: return np.array([])
    
    # Optimierung: Pre-Allocation der Dicts reduziert Re-Hashing
    adj = {}
    
    # Iteration über Numpy Array ist schnell genug für Boundary-Größe (meist < 100k)
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    nodes = list(adj.keys())
    if not nodes: return np.array([])
    
    # Greedy Pfadsuche
    curr, path, visited = nodes[0], [nodes[0]], {nodes[0]}

    while True:
        neighbors = adj.get(curr, [])
        next_node = None
        for n in neighbors:
            if n not in visited:
                next_node = n; break
        if next_node is None:
            # Check ob Loop geschlossen zum Startknoten
            for n in neighbors:
                if n == path[0] and len(path) > 2: return np.array(path)
            break
        path.append(next_node); visited.add(next_node); curr = next_node
        
    return np.array(path)

def _generate_base_and_walls_for_mesh(ply_path, output_folder, boundary_type='complex'):
    """
    Streaming-Architektur V2: Seek & Update + Low-Mem Boundary Calc.
    """
    try:
        CHUNK_SIZE = 50000 
        
        stl_dtype = np.dtype([
            ('normal', '<f4', (3,)),
            ('v1',     '<f4', (3,)),
            ('v2',     '<f4', (3,)),
            ('v3',     '<f4', (3,)),
            ('attr',   '<u2')
        ])

        out_name = os.path.basename(ply_path).replace('.ply', '_walls.stl')
        out_path = os.path.join(output_folder, out_name)
        total_triangles_written = 0

        with open(out_path, 'wb') as f:
            # Header Placeholder
            header_placeholder = b'Binary STL (Streamed) - Header will be updated...'.ljust(80, b'\0')
            f.write(header_placeholder)
            f.write(struct.pack('<I', 0))

            chunk_buffer = np.zeros(CHUNK_SIZE, dtype=stl_dtype)

            # --- SCHRITT 1: Original Mesh ---
            mesh = trimesh.load(ply_path, process=False)
            if mesh.vertices.dtype != np.float32:
                mesh.vertices = mesh.vertices.astype(np.float32)

            verts = mesh.vertices
            z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
            floor_z = np.float32(z_min - (z_max - z_min) * 0.1)

            # --- Pfad-Logik (Optimiert) ---
            if boundary_type == 'rectangular':
                # (Rechteckige Logik unverändert, da sie bereits effizient ist)
                x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
                y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
                tol = 1e-3
                idx_x_min = np.where(np.abs(verts[:, 0] - x_min) < tol)[0]
                idx_x_max = np.where(np.abs(verts[:, 0] - x_max) < tol)[0]
                idx_y_min = np.where(np.abs(verts[:, 1] - y_min) < tol)[0]
                idx_y_max = np.where(np.abs(verts[:, 1] - y_max) < tol)[0]
                
                s1 = verts[idx_x_min]; idx_s1 = idx_x_min[np.lexsort((s1[:, 1],))]
                s2 = verts[idx_y_max]; idx_s2 = idx_y_max[np.lexsort((s2[:, 0],))]
                s3 = verts[idx_x_max]; idx_s3 = idx_x_max[np.lexsort((s3[:, 1],))][::-1]
                s4 = verts[idx_y_min]; idx_s4 = idx_y_min[np.lexsort((s4[:, 0],))][::-1]
                path_indices = np.concatenate([idx_s1, idx_s2, idx_s3, idx_s4])
                path_indices = path_indices[np.sort(np.unique(path_indices, return_index=True)[1])]
            else:
                # HIER IST DER FIX: Nutze die speicheroptimierte Funktion statt trimesh Properties
                boundary_edges = extract_boundary_edges_memory_efficiently(mesh)
                path_indices = _sort_edges_into_closed_path(boundary_edges)
                
                # Cleanup immediate
                del boundary_edges
                gc.collect()

            if len(path_indices) < 3: 
                print("Warnung: Kein geschlossener Rand gefunden.")
                return False

            top_ring_verts = mesh.vertices[path_indices]
            
            # --- Stream Original Mesh ---
            num_faces = len(mesh.faces)
            for i in range(0, num_faces, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, num_faces)
                count = end - i
                
                face_chunk = mesh.faces[i:end]
                v1 = mesh.vertices[face_chunk[:, 0]]
                v2 = mesh.vertices[face_chunk[:, 1]]
                v3 = mesh.vertices[face_chunk[:, 2]]
                
                norms = np.cross(v2 - v1, v3 - v1)
                norm_len = np.linalg.norm(norms, axis=1, keepdims=True)
                norm_len[norm_len == 0] = 1.0
                norms /= norm_len

                target = chunk_buffer[:count]
                target['normal'] = norms
                target['v1'], target['v2'], target['v3'] = v1, v2, v3
                target['attr'] = 0
                
                f.write(target.tobytes())
                total_triangles_written += count

            del mesh, verts
            gc.collect()

            # --- SCHRITT 2: Wände ---
            n_path = len(top_ring_verts)
            curr_idx = np.arange(n_path)
            next_idx = (curr_idx + 1) % n_path
            
            v_top_curr = top_ring_verts[curr_idx]
            v_top_next = top_ring_verts[next_idx]
            v_bot_curr = v_top_curr.copy(); v_bot_curr[:, 2] = floor_z
            v_bot_next = v_top_next.copy(); v_bot_next[:, 2] = floor_z
            
            w_v1 = np.vstack([v_top_curr, v_top_next])
            w_v2 = np.vstack([v_bot_curr, v_bot_curr])
            w_v3 = np.vstack([v_top_next, v_bot_next])
            
            wall_count = len(w_v1)
            
            for i in range(0, wall_count, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, wall_count)
                count = end - i
                
                wv1, wv2, wv3 = w_v1[i:end], w_v2[i:end], w_v3[i:end]
                wnorms = np.cross(wv2 - wv1, wv3 - wv1)
                wlen = np.linalg.norm(wnorms, axis=1, keepdims=True)
                wlen[wlen == 0] = 1.0
                wnorms /= wlen
                
                target = chunk_buffer[:count]
                target['normal'] = wnorms
                target['v1'], target['v2'], target['v3'] = wv1, wv2, wv3
                target['attr'] = 0
                f.write(target.tobytes())
                total_triangles_written += count
            
            del w_v1, w_v2, w_v3, v_top_curr, v_bot_curr
            gc.collect()

            # --- SCHRITT 3: Boden ---
            poly = Polygon(top_ring_verts[:, :2])
            if not poly.is_valid: 
                poly = poly.buffer(0)
            
            # FIX 1: Shapely's buffer(0) erzeugt bei Selbstüberschneidungen oft ein MultiPolygon.
            # Trimesh erwartet aber ein einzelnes Polygon. Wir filtern das Haupt-Polygon (größte Fläche) heraus.
            if getattr(poly, 'geom_type', '') == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda a: a.area)
            
            tri_result = trimesh.creation.triangulate_polygon(poly)
            
            # FIX 2: Absicherung gegen leere Tuples, falls die Triangulierung komplett abbricht
            if isinstance(tri_result, trimesh.Trimesh):
                f_verts, f_faces = tri_result.vertices, tri_result.faces
            elif isinstance(tri_result, tuple) and len(tri_result) >= 2:
                f_verts = np.array(tri_result[0])
                f_faces = np.array(tri_result[1])
            else:
                # Notfall-Fallback: Kein Boden, aber das Programm stürzt nicht ab
                f_verts = np.empty((0, 3), dtype=np.float32)
                f_faces = np.empty((0, 3), dtype=np.int32)
            
            if len(f_verts) > 0 and f_verts.shape[1] == 2:
                zeros = np.zeros((len(f_verts), 1), dtype=np.float32)
                f_verts = np.hstack((f_verts, zeros))
                
            if len(f_verts) > 0:
                f_verts[:, 2] = floor_z
            
            if len(f_faces) > 0:
                t0 = f_verts[f_faces[0]]
                if np.cross(t0[1]-t0[0], t0[2]-t0[0])[2] > 0:
                    f_faces = f_faces[:, ::-1]

            num_floor_faces = len(f_faces)
            for i in range(0, num_floor_faces, CHUNK_SIZE):
                end = min(i + CHUNK_SIZE, num_floor_faces)
                count = end - i
                
                f_chunk = f_faces[i:end]
                fv1 = f_verts[f_chunk[:, 0]]
                fv2 = f_verts[f_chunk[:, 1]]
                fv3 = f_verts[f_chunk[:, 2]]
                
                fnorms = np.cross(fv2 - fv1, fv3 - fv1)
                flen = np.linalg.norm(fnorms, axis=1, keepdims=True)
                flen[flen == 0] = 1.0
                fnorms /= flen
                
                target = chunk_buffer[:count]
                target['normal'] = fnorms
                target['v1'], target['v2'], target['v3'] = fv1, fv2, fv3
                target['attr'] = 0
                f.write(target.tobytes())
                total_triangles_written += count

            del f_verts, f_faces, top_ring_verts
            gc.collect()

            # --- Header Update ---
            f.seek(80)
            f.write(struct.pack('<I', total_triangles_written))
            
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Memory/Computation Error: {e}")
        return False

def batch_generate_mesh_enclosures(input_file_path, output_folder, boundary_type='complex'):
    """
    Extruhiert Wände und einen Boden für alle Meshes in einem Ordner.
    Macht offene Geländemodelle 'wasserdicht' und druckbar (3D-Druck).

    Args:
        input_file_path (str): Pfad zur .ply Einzeldatei.
        output_folder (str): Zielordner für .stl Dateien.
        boundary_type (str, optional):
            'rectangular': Erzwingt eine rechteckige Box (gut für Kacheln).
            'complex': Folgt dem exakten Rand des Meshes (gut für organische Scans).
            Defaults to 'complex'.
    """
    if not os.path.isfile(input_file_path):
        print(f"Fehler: '{input_file_path}' ist keine gültige Datei.", flush=True)
        return

    os.makedirs(output_folder, exist_ok=True)
    print(f"Erstelle Wände für: {os.path.basename(input_file_path)}...", flush=True)
    
    success = _generate_base_and_walls_for_mesh(input_file_path, output_folder, boundary_type)
    
    if success:
        print(f"Fertig. Gespeichert in: {output_folder}", flush=True)
    else:
        print("Fehler: Wände konnten nicht generiert werden.", flush=True)
