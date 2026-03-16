# src/core/processing.py

import os
import multiprocessing

import numpy as np
from tqdm import tqdm
import scipy.ndimage
from scipy.stats import binned_statistic_2d
from pyproj import Transformer

from .utils import _save_data_to_npz, extract_point_cloud_from_npz, _collect_npz_file_paths


def remove_morphological_artifacts(z_data, kernel_size):
    """
    Entfernt dünne, hohe Strukturen (z.B. Baukräne) mittels Morphological Opening.
    Behandelt NaN-Werte robust, um Randeffekte zu vermeiden.

    Args:
        z_data (np.ndarray): 2D-Array (Float32) mit Höhendaten.
        kernel_size (int): Größe des Strukturelements (in Pixeln).
                           Sollte größer sein als die Breite der Störstruktur.

    Returns:
        np.ndarray: Bereinigtes Z-Grid.
    """
    # 1. Check: Wenn alles NaN ist, sofort zurück
    if np.all(np.isnan(z_data)):
        return z_data

    # 2. NaN-Maske speichern
    nan_mask = np.isnan(z_data)

    # 3. NaNs füllen für die Operation
    # Wir füllen mit dem globalen Minimum der validen Daten.
    # Grund: Opening = Erosion (Min) -> Dilation (Max).
    # Wenn wir NaNs sehr klein machen, "gewinnt" bei der Erosion das Loch (korrekt),
    # und bei der Dilation wird es nicht künstlich hochgezogen.
    filled_data = z_data.copy()
    valid_min = np.nanmin(z_data)
    filled_data[nan_mask] = valid_min

    # 4. Strukturelement definieren (Disk statt Quadrat für Rotationsinvarianz empfohlen)
    # Ein quadratischer Kernel (np.ones) würde bei diagonalen Kränen versagen.
    radius = kernel_size // 2
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    structure = (x**2 + y**2) <= radius**2

    # 5. Morphological Opening anwenden (Entfernt helle/hohe Strukturen kleiner als Kernel)
    # output=filled_data spart Speicher
    scipy.ndimage.grey_opening(filled_data, structure=structure, output=filled_data)

    # 6. NaNs wiederherstellen
    filled_data[nan_mask] = np.nan

    return filled_data

def crop_dataset_to_bounds(input_folder, output_folder, bounds, data_crs=None, bounds_crs=None):
    """Schneidet alle Datensätze auf eine definierte Bounding Box zu.

    Funktioniert sowohl für Grid- als auch für Raw-Daten.
    Kann optional Koordinatensysteme transformieren, falls Bounds und Daten abweichen.

    Args:
        input_folder (str): Quellordner.
        output_folder (str): Zielordner.
        bounds (tuple): (xmin, xmax, ymin, ymax).
        data_crs (str, optional): CRS der Daten (z.B. "EPSG:25832").
        bounds_crs (str, optional): CRS der Bounds (z.B. "EPSG:4326" für GPS).
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]
    xmin, xmax, ymin, ymax = bounds

    if data_crs is not None and bounds_crs is not None and data_crs != bounds_crs:
        print(f"Transformiere Bounds von {bounds_crs} nach {data_crs}...", flush=True)
        transformer = Transformer.from_crs(bounds_crs, data_crs, always_xy=True)
        xmin, ymin = transformer.transform(xmin, ymin)
        xmax, ymax = transformer.transform(xmax, ymax)
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin
        print(f"Neue Bounds: X[{xmin:.1f}, {xmax:.1f}], Y[{ymin:.1f}, {ymax:.1f}]", flush=True)

    for filename in tqdm(files, desc="Zuschneiden"):
        filepath = os.path.join(input_folder, filename)
        try:
            loaded = np.load(filepath, allow_pickle=True)
            dtype = str(loaded['type'])

            attributes = None
            if 'attributes' in loaded:
                try: attributes = loaded['attributes'].item()
                except: pass

            if dtype == 'grid':
                z = loaded['data']; meta = loaded['meta']
                colors = loaded['colors'] if 'colors' in loaded else None
                x_off, y_off, res_x, res_y = meta
                rows, cols = z.shape

                # Unabhängig vom Vorzeichen der Auflösung min/max bestimmen
                ex_x = sorted([x_off, x_off + cols * res_x])
                ex_y = sorted([y_off, y_off + rows * res_y])
                
                g_x_min, g_x_max = ex_x[0], ex_x[1]
                g_y_min, g_y_max = ex_y[0], ex_y[1]

                # INTERSECTION CHECK (Robust gegen negative Res)
                # Wenn Grid komplett außerhalb der Crop-Box liegt -> Skip
                if (g_x_max < xmin) or (g_x_min > xmax) or (g_y_max < ymin) or (g_y_min > ymax):
                    continue
                
                col_start = int(np.clip((xmin - x_off) / res_x, 0, cols))
                col_end   = int(np.clip((xmax - x_off) / res_x, 0, cols))
                row_start = int(np.clip((ymin - y_off) / res_y, 0, rows))
                row_end   = int(np.clip((ymax - y_off) / res_y, 0, rows))
                
                # Falls res negativ ist, ist start > end. Das korrigieren wir hier.
                c_s, c_e = sorted([col_start, col_end])
                r_s, r_e = sorted([row_start, row_end])

                # Leere Slices überspringen
                if c_s >= c_e or r_s >= r_e: 
                    continue

                # DATEN KOPIEREN
                z_crop = z[r_s:r_e, c_s:c_e]
                colors_crop = colors[r_s:r_e, c_s:c_e] if colors is not None else None
                
                # METADATEN UPDATE
                # Neuer Origin ist abhängig von der Slice-Richtung
                # Bei negativer Res: y_off + r_s * res_y ist immer noch korrekt (man wandert "nach unten")
                new_meta = np.array([
                    x_off + c_s * res_x, 
                    y_off + r_s * res_y, 
                    res_x, 
                    res_y
                ])

                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(output_folder, f"{base_name}_cropped.npz")
                _save_data_to_npz(save_path, z_crop, new_meta, 'grid', colors_crop, attributes=attributes)

            elif dtype == 'raw':
                points = loaded['data']
                colors = loaded['colors'] if 'colors' in loaded else None
                normals = loaded['normals'] if 'normals' in loaded else None

                mask = ((points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax))
                if not np.any(mask): continue

                attributes_crop = None
                if attributes is not None:
                    attributes_crop = {}
                    for k, v in attributes.items():
                        if isinstance(v, (np.ndarray, list)) and len(v) == len(points):
                            attributes_crop[k] = v[mask]
                        else:
                            attributes_crop[k] = v

                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(output_folder, f"{base_name}_cropped.npz")

                _save_data_to_npz(
                    save_path, points[mask],
                    data_type='raw',
                    colors=colors[mask] if colors is not None else None,
                    normals=normals[mask] if normals is not None else None,
                    attributes=attributes_crop
                )
        except Exception as e:
            print(f"Fehler bei {filename}: {e}", flush=True)

def scale_elevation_in_dataset(input_folder, output_folder, factor):
    """Skaliert die Z-Werte (Höhe) aller Datensätze.

    Nützlich zur Überhöhung von Gelände für 3D-Druck oder Visualisierung.

    Args:
        input_folder (str): Quellordner.
        output_folder (str): Zielordner.
        factor (float): Multiplikationsfaktor für die Höhe (z.B. 2.0).
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]

    for filename in tqdm(files, desc=f"Z-Stretch x{factor}"):
        filepath = os.path.join(input_folder, filename)
        save_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_stretched.npz")

        try:
            loaded = np.load(filepath, allow_pickle=True)
            data = loaded['data'].copy()
            save_dict = {k: loaded[k] for k in loaded.files}

            if str(loaded['type']) == 'grid':
                save_dict['data'] = data * factor
            elif str(loaded['type']) == 'raw':
                data[:, 2] = data[:, 2] * factor
                save_dict['data'] = data
                if 'normals' in save_dict: del save_dict['normals']

            np.savez_compressed(save_path, **save_dict)
        except Exception as e:
            print(f"Fehler bei {filename}: {e}", flush=True)
            
def rasterize_point_cloud_to_grid(input_folder, output_folder, resolution=1.0):
    """
    Rasterisiert RAW-Punktwolken zurück in ein Grid.
    Führt ein Re-Binning durch.

    Args:
        input_folder (str): Quellordner mit .npz Dateien (Typ: raw).
        output_folder (str): Zielordner.
        resolution (float): Gewünschte Rasterweite in Metern (z.B. 1.0). Gute wahl ist die ursprüngliche Auflösung + 2 zu wählen.
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]

    for filename in tqdm(files, desc="Raw -> Grid (Re-Binning)"):
        filepath = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            loaded = np.load(filepath, allow_pickle=True)
            if str(loaded['type']) != 'raw':
                print(f"Überspringe {filename}: Ist kein Raw-Datensatz.", flush=True)
                continue

            # Daten extrahieren
            data = loaded['data'] # (N, 3)
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            
            colors = loaded['colors'] if 'colors' in loaded else None
            attributes = loaded['attributes'].item() if 'attributes' in loaded else {}

            # Grid Dimensionen berechnen
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            
            # Puffer, um Randeffekte zu vermeiden
            cols = int(np.ceil((x_max - x_min) / resolution)) + 1
            rows = int(np.ceil((y_max - y_min) / resolution)) + 1

            x_edges = np.linspace(x_min, x_min + cols * resolution, cols + 1)
            y_edges = np.linspace(y_min, y_min + rows * resolution, rows + 1)

            # 1. Z-Werte Rastern (Mittelwert pro Zelle)
            z_grid, _, _, _ = binned_statistic_2d(
                x, y, z, 
                statistic='mean', 
                bins=[x_edges, y_edges]
            )
            # Transponieren für Bild-Koordinaten (Rows, Cols)
            z_grid = z_grid.T 

            # 2. Farben Rastern (falls vorhanden)
            colors_grid = None
            if colors is not None:
                r_grid = binned_statistic_2d(x, y, colors[:,0], statistic='mean', bins=[x_edges, y_edges])[0].T
                g_grid = binned_statistic_2d(x, y, colors[:,1], statistic='mean', bins=[x_edges, y_edges])[0].T
                b_grid = binned_statistic_2d(x, y, colors[:,2], statistic='mean', bins=[x_edges, y_edges])[0].T
                colors_grid = np.dstack((r_grid, g_grid, b_grid)).astype(np.float32)

            # 3. Metadaten erstellen
            # Ursprung ist die untere linke Ecke des Rasters (x_min, y_min)
            meta = np.array([x_min, y_min, resolution, resolution])
            
            # Attribute aktualisieren
            attributes['processing_history'] = attributes.get('processing_history', "") + " -> Re-Gridded"

            save_path = os.path.join(output_folder, f"{base_name}_regridded.npz")
            _save_data_to_npz(save_path, z_grid.astype(np.float64), meta, 'grid', colors=colors_grid, attributes=attributes)

        except Exception as e:
            print(f"Fehler bei {filename}: {e}", flush=True)            

def merge_npz_datasets(input_path, output_path):
    """Fügt mehrere Kacheln zu einer einzigen großen Datei zusammen.

    Unterstützt intelligentes Stitching von Grid-Daten (Umgang mit Überlappungen)
    und Concatenation von Raw-Punktwolken. Übernimmt Attribute und Farben.

    Args:
        input_path (str): Ordner mit den Kacheln.
        output_path (str): Pfad der Ausgabedatei (.npz).
    """
    files = _collect_npz_file_paths(input_path)
    if not files: return

    print(f"Referenzdatei: {os.path.basename(files[0])}", flush=True)
    try:
        ref_loaded = np.load(files[0], allow_pickle=True)
        ref_type = str(ref_loaded['type'])
        ref_meta = ref_loaded['meta'] if 'meta' in ref_loaded else None
        ref_has_color = 'colors' in ref_loaded
        ref_res_x = ref_meta[2] if ref_meta is not None else None
        ref_res_y = ref_meta[3] if ref_meta is not None else None

        ref_attributes = None
        if 'attributes' in ref_loaded:
            try: ref_attributes = ref_loaded['attributes'].item()
            except: pass

    except Exception as e:
        print(f"Fehler Referenzdatei: {e}", flush=True); return

    valid_files_data = []
    for f in tqdm(files, desc="Validierung"):
        try:
            curr = np.load(f, allow_pickle=True)
            if str(curr['type']) == ref_type:
                if ref_type == 'grid':
                    rows, cols = curr['data'].shape
                    x_off, y_off = curr['meta'][0], curr['meta'][1]
                    
                    # FIX: Sortierung fängt negative Auflösungen ab
                    ex_x = sorted([x_off, x_off + cols * ref_res_x])
                    ex_y = sorted([y_off, y_off + rows * ref_res_y])
                    
                    valid_files_data.append({
                        'path': f, 'meta': curr['meta'], 'rows': rows, 'cols': cols,
                        'x_min': ex_x[0], 'x_max': ex_x[1],
                        'y_min': ex_y[0], 'y_max': ex_y[1]
                    })
                else:
                    valid_files_data.append({'path': f})
        except: pass

    if not valid_files_data: return
    print(f"Merge von {len(valid_files_data)} Dateien...", flush=True)

    if ref_type == 'raw':
        list_xyz, list_rgb, list_norm = [], [], []
        raw_keys = [k for k in ref_attributes.keys() if isinstance(ref_attributes[k], (np.ndarray, list))] if ref_attributes else []
        list_attrs = {k: [] for k in raw_keys}

        for item in tqdm(valid_files_data, desc="Merging Raw"):
            l = np.load(item['path'], allow_pickle=True)
            list_xyz.append(l['data'])
            if ref_has_color and 'colors' in l: list_rgb.append(l['colors'])
            if 'normals' in l: list_norm.append(l['normals'])

            if raw_keys and 'attributes' in l:
                try:
                    curr_attrs = l['attributes'].item()
                    for k in raw_keys:
                        if k in curr_attrs: list_attrs[k].append(curr_attrs[k])
                except: pass

        full_xyz = np.vstack(list_xyz)
        full_rgb = np.vstack(list_rgb) if list_rgb else None
        full_norm = np.vstack(list_norm) if list_norm else None

        full_attrs = None
        if ref_attributes:
            full_attrs = ref_attributes.copy()
            for k in raw_keys:
                if list_attrs[k]: full_attrs[k] = np.concatenate(list_attrs[k])

        _save_data_to_npz(output_path, full_xyz, data_type='raw', colors=full_rgb, normals=full_norm, attributes=full_attrs)

    elif ref_type == 'grid':
        g_x_min = min(d['x_min'] for d in valid_files_data)
        g_x_max = max(d['x_max'] for d in valid_files_data)
        g_y_min = min(d['y_min'] for d in valid_files_data)
        g_y_max = max(d['y_max'] for d in valid_files_data)

        # FIX 1: Absolute Werte für Array-Dimensionen
        total_cols = int(np.ceil((g_x_max - g_x_min) / abs(ref_res_x)))
        total_rows = int(np.ceil((g_y_max - g_y_min) / abs(ref_res_y)))

        full_grid = np.full((total_rows, total_cols), np.nan, dtype=np.float32)
        full_colors = np.full((total_rows, total_cols, 3), 0.0, dtype=np.float32) if ref_has_color else None

        # FIX 2: Der Ursprung ist bei negativer Y-Auflösung "oben" (Max Y)
        new_x_off = g_x_min if ref_res_x > 0 else g_x_max
        new_y_off = g_y_min if ref_res_y > 0 else g_y_max

        for item in tqdm(valid_files_data, desc="Stitching"):
            loaded = np.load(item['path'], allow_pickle=True)
            z_chunk = loaded['data']
            c_chunk = loaded['colors'] if ref_has_color else None
            meta_chunk = loaded['meta']

            # FIX 3: Berechnung relativ zum dynamischen Ursprung
            col_start = int(round((meta_chunk[0] - new_x_off) / ref_res_x))
            row_start = int(round((meta_chunk[1] - new_y_off) / ref_res_y))
            
            chunk_rows, chunk_cols = z_chunk.shape
            row_end = min(row_start + chunk_rows, total_rows)
            col_end = min(col_start + chunk_cols, total_cols)
            eff_rows, eff_cols = row_end - row_start, col_end - col_start

            if eff_rows > 0 and eff_cols > 0:
                target = full_grid[row_start:row_end, col_start:col_end]
                src = z_chunk[:eff_rows, :eff_cols]
                mask = ~np.isnan(src)
                target[mask] = src[mask]
                full_grid[row_start:row_end, col_start:col_end] = target
                if full_colors is not None and c_chunk is not None:
                     full_colors[row_start:row_end, col_start:col_end][mask] = c_chunk[:eff_rows, :eff_cols][mask]

        # FIX 4: Neue Metadaten mit dem korrekten Ursprung
        new_meta = np.array([new_x_off, new_y_off, ref_res_x, ref_res_y])
        _save_data_to_npz(output_path, full_grid, new_meta, 'grid', full_colors, attributes=ref_attributes)
    print(f"Gespeichert: {output_path}", flush=True)

def transform_coordinate_reference_system(data, src_crs="EPSG:25832", tgt_crs="EPSG:3035"):
    """
    Transformiert ein NumPy-Array von Koordinaten zwischen zwei Koordinatensystemen.
    
    Robust und Low-Level: Akzeptiert (N,2) oder (N,3) Arrays.

    Args:
        data (np.ndarray): Array der Form (N, 2) [x, y] oder (N, 3) [x, y, z].
        src_crs (str): Quell-CRS (z.B. "EPSG:25832").
        tgt_crs (str): Ziel-CRS (z.B. "EPSG:3035").

    Returns:
        np.ndarray: Transformiertes Array mit gleicher Dimension wie Input.
        
    Raises:
        ValueError: Wenn das Input-Format falsch ist.
    """
    if data is None or len(data) == 0:
        return data

    cols = data.shape[1]
    if cols < 2 or cols > 3:
        raise ValueError(f"Input Data muss 2 (XY) oder 3 (XYZ) Spalten haben. Hat: {cols}")

    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
    
    # Trennung der Koordinaten für pyproj
    x, y = data[:, 0], data[:, 1]
    z = data[:, 2] if cols == 3 else None

    if z is not None:
        # 3D Transformation (Falls Höhensystem-Wechsel definiert ist im CRS)
        tx, ty, tz = transformer.transform(x, y, z)
        return np.column_stack((tx, ty, tz))
    else:
        # 2D Transformation
        tx, ty = transformer.transform(x, y)
        return np.column_stack((tx, ty))

def _process_crs_transformation_task(file_path, output_folder, src_crs, tgt_crs, grid_mode):
    """Worker-Funktion für batch_transform_dataset_crs (Multiprocessing).
    
    Muss Top-Level definiert sein, um picklable zu sein.
    Initialisiert den Transformer lokal pro Prozess.
    """
    try:
        # Transformer lokal erstellen (Thread-Safe für Multiprocessing)
        # pyproj Objekte lassen sich oft schlecht picklen, daher Neuerstellung im Worker.
        transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        loaded = np.load(file_path, allow_pickle=True)
        dtype = str(loaded['type'])
        
        # Attribute laden & CRS Tag updaten
        attributes = {}
        if 'attributes' in loaded:
            try: 
                attributes = loaded['attributes'].item()
            except: pass
        attributes['crs'] = tgt_crs
        attributes['src_crs_history'] = f"{src_crs} -> {tgt_crs}"

        # --- FALL 1: RAW DATEN (PUNKTWOLKE) ---
        if dtype == 'raw':
            data = loaded['data'] # (N, 3)
            
            # Transformation
            tx, ty, tz = transformer.transform(data[:, 0], data[:, 1], data[:, 2])
            new_data = np.column_stack((tx, ty, tz)).astype(np.float32)
            
            save_path = os.path.join(output_folder, f"{base_name}_trans.npz")
            _save_data_to_npz(
                save_path, new_data, data_type='raw',
                colors=loaded['colors'] if 'colors' in loaded else None,
                normals=loaded['normals'] if 'normals' in loaded else None, 
                attributes=attributes
            )

        # --- FALL 2: GRID DATEN (RASTER) ---
        elif dtype == 'grid':
            if grid_mode == 'exact':
                # Strategie: Grid -> Raw (Punktwolke)
                xyz, rgb = extract_point_cloud_from_npz(file_path)
                
                tx, ty, tz = transformer.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2])
                new_data = np.column_stack((tx, ty, tz)).astype(np.float32)
                
                save_path = os.path.join(output_folder, f"{base_name}_grid2raw_trans.npz")
                _save_data_to_npz(
                    save_path, new_data, data_type='raw', # Typ ändert sich!
                    colors=rgb,
                    attributes=attributes
                )
                
            elif grid_mode == 'fast':
                # Strategie: Nur Origin shiften
                z_data = loaded['data']
                meta = loaded['meta'] # [x, y, res_x, res_y]
                
                # Transformiere den Ursprung (untere linke Ecke)
                tx, ty = transformer.transform(meta[0], meta[1])
                
                new_meta = np.array([tx, ty, meta[2], meta[3]])
                
                save_path = os.path.join(output_folder, f"{base_name}_trans.npz")
                _save_data_to_npz(
                    save_path, z_data, new_meta, 'grid',
                    colors=loaded['colors'] if 'colors' in loaded else None,
                    attributes=attributes
                )
        return True
    except Exception as e:
        print(f"Fehler in Worker für {file_path}: {e}", flush=True)
        return False

def batch_transform_dataset_crs(input_source, output_folder, src_crs, tgt_crs, grid_mode='exact', debug=True):
    """
    Batch-Transformation für .npz Datensätze (Raw & Grid) in ein neues Koordinatensystem.

    Features:
    - Verarbeitet einzelne Dateien oder ganze Ordner.
    - Multiprocessing-Support für hohe Geschwindigkeit.
    - Unterscheidet Raw (Punktwolke) und Grid (Raster).
    
    Args:
        input_source (str): Pfad zu einer .npz Datei oder einem Ordner mit .npz Dateien.
        output_folder (str): Zielordner für die transformierten Dateien.
        src_crs (str): Quell-Koordinatensystem (z.B. "EPSG:25832").
        tgt_crs (str): Ziel-Koordinatensystem (z.B. "EPSG:4326" oder "EPSG:3035").
        grid_mode (str, optional): Strategie für Grid-Daten.
            - 'exact': (Empfohlen) Konvertiert Grid -> Raw (Punktwolke).
              Garantiert positionsgetreue Transformation jedes Pixels (Rotation!), ändert aber Datentyp.
            - 'fast': Transformiert nur den Ursprung. Hält Daten als Grid.
              ACHTUNG: Nur bei reiner Translation (keine Rotation!) verwenden.
            Defaults to 'exact'.
        debug (bool, optional): 
            - True: Sequentielle Ausführung (Single-Core) für Fehlersuche.
            - False: Parallele Ausführung (Multi-Core).
            Defaults to True.
    """
    # 1. Dateien sammeln
    files = []
    if os.path.isfile(input_source) and input_source.endswith('.npz'):
        files = [input_source]
    elif os.path.isdir(input_source):
        files = [os.path.join(input_source, f) for f in os.listdir(input_source) if f.endswith('.npz')]
    
    if not files:
        print("Keine .npz Dateien gefunden.", flush=True)
        return

    os.makedirs(output_folder, exist_ok=True)
    num_files = len(files)
    
    # Argumente für jeden Worker vorbereiten
    args = [(f, output_folder, src_crs, tgt_crs, grid_mode) for f in files]

    print(f"Transformiere {num_files} Dateien: {src_crs} -> {tgt_crs} (Mode: {grid_mode})", flush=True)

    if debug:
        print("DEBUG MODE: Sequentielle Ausführung...", flush=True)
        for arg in tqdm(args, desc="CRS Transform (Single)"):
            _process_crs_transformation_task(*arg)
    else:
        # Anzahl Kerne begrenzen auf Anzahl Dateien (vermeidet Overhead)
        cpu_cores = multiprocessing.cpu_count()
        num_processes = max(1, min(num_files, cpu_cores))
        
        print(f"Starte {num_processes} Prozesse...", flush=True)
        
        # Pool starten
        with multiprocessing.Pool(processes=num_processes) as pool:
            # starmap entpackt das args-Tupel automatisch in die Funktionsargumente
            list(tqdm(pool.starmap(_process_crs_transformation_task, args), total=num_files, desc="CRS Transform (Multi)"))

def transform_bavarian_to_austrian_crs(data):
    """Wrapper für EPSG:25832 (Bayern) nach EPSG:3035 (Österreich) inkl. Achsentausch.

    Österreichische Systeme nutzen oft vertauschte X/Y Achsen im Vergleich zu Standard-UTM.

    Args:
        data (np.ndarray): Input Punkte (N, 3).

    Returns:
        np.ndarray: Transformierte Punkte.
    """
    # 1. Standard Transformation
    transformed = transform_coordinate_reference_system(data, "EPSG:25832", "EPSG:3035")
    
    # 2. Achsentausch (Swap X/Y)
    # Numpy Slicing: create new array to ensure contiguous memory
    final = np.empty_like(transformed)
    final[:, 0] = transformed[:, 1] # New X = Old Y
    final[:, 1] = transformed[:, 0] # New Y = Old X
    final[:, 2] = transformed[:, 2] # Z bleibt
    
    return final

def transform_austrian_to_bavarian_crs(data):
    """Wrapper für EPSG:3035 nach EPSG:25832 inkl. Achsenrücktausch.

    Args:
        data (np.ndarray): Input Punkte (N, 3) in AT-Format (Y, X, Z).

    Returns:
        np.ndarray: Transformierte Punkte in DE-Format.
    """
    # 1. Rück-Tausch der Input-Achsen, damit pyproj Standard (X, Y) bekommt
    # Input ist [Y_proj, X_proj, Z], wir brauchen [X_proj, Y_proj, Z] für den Transformer
    x_in = data[:, 1]
    y_in = data[:, 0]
    z_in = data[:, 2] if data.shape[1] > 2 else None
    
    temp_data = np.column_stack((x_in, y_in, z_in)) if z_in is not None else np.column_stack((x_in, y_in))

    # 2. Standard Transformation
    return transform_coordinate_reference_system(temp_data, "EPSG:3035", "EPSG:25832")
