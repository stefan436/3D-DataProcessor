# src/core/io_operations.py

import os
import zipfile

import numpy as np
from tqdm import tqdm
import laspy
import rasterio
from rasterio.enums import Resampling
from scipy.stats import binned_statistic_2d

from .utils import _remove_invalid_data_points
from .utils import _save_data_to_npz
from .processing import remove_morphological_artifacts
from config.settings import NODATA_VALUE


def convert_geotiff_to_npz(input_folder, output_folder, kranfilter=False, kernel_size=6, target_resolution=None):
    """Konvertiert GeoTIFF-Rasterdaten in das Grid-NPZ Format.

    Ersetzt NODATA Werte explizit durch np.nan, um Verzerrungen zu vermeiden.

    Args:
        input_folder (str): Pfad zum Ordner mit GeoTIFF-Dateien.
        output_folder (str): Zielordner für die .npz Dateien.
        Kranfilter (bool, optional): Ob ein Filter der Kräne in DOMs herausfiltert Aktiviert werden soll. Über Gaus'schen Kernel. Filtert Teils auch Antennen.
        Defaults to False.
        kernel_size (int): Anzahl der Pixel in der Umgebung die Relevant sind für den Filter. Hohe Werte filtern mehr. Gute range ist 5-7.
        Defaults to 6.
        target_resolution (float, optional): Ziel-Auflösung in Metern. 
                                             Ist dieser Wert None (Standard), werden die Daten 1:1 eingelesen (kein Binning).
    """
    os.makedirs(output_folder, exist_ok=True)
    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]

    for filename in tqdm(tiff_files, desc="GeoTIFF -> NPZ"):
        filepath = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_folder, f"{base_name}_grid.npz")

        try:
            with rasterio.open(filepath) as src:
                # 1. Aktuelle Auflösung ermitteln (positiver Wert der Pixelbreite)
                current_res = abs(src.transform.a) 
                
                # --- ENTSCHEIDUNG: BINNING ODER RAW READ ---
                # Binning nur aktivieren, wenn target_resolution gesetzt ist UND größer als aktuelle Auflösung ist
                if target_resolution is not None and target_resolution > current_res:
                    scale_factor = current_res / target_resolution
                    
                    # Neue Dimensionen berechnen
                    new_width = int(src.width * scale_factor)
                    new_height = int(src.height * scale_factor)
                    
                    # Lesen mit 'Average' Resampling (das entspricht dem Binning)
                    z_data = src.read(
                        1,
                        out_shape=(new_height, new_width),
                        resampling=Resampling.average
                    )
                    
                    # Transformation anpassen (Pixel werden größer, Bild wird kleiner)
                    t = src.transform * src.transform.scale(
                        (src.width / new_width),
                        (src.height / new_height)
                    )
                    
                    actual_res = target_resolution
                else:
                    # --- DEAKTIVIERTES BINNING (Standard) ---
                    # Liest die Daten in voller Original-Auflösung
                    z_data = src.read(1)
                    t = src.transform
                    actual_res = current_res
                
                # --- AB HIER: NORMALE VERARBEITUNG ---
                # Explizite Behandlung von Source-Nodata UND globalem NODATA_VALUE
                src_nodata = src.nodata
                
                # Zuerst auf Float32 casten, damit wir np.nan setzen können
                if z_data.dtype != np.float32: 
                    z_data = z_data.astype(np.float32)

                # Maskierung anwenden
                if src_nodata is not None:
                    z_data[np.isclose(z_data, src_nodata)] = np.nan
                
                # Zusätzlich globalen NODATA checken
                z_data[np.isclose(z_data, NODATA_VALUE)] = np.nan
                
                if kranfilter:
                    # Kräne sind ca. 2-4 px breit -> Kernelgröße 5 oder 7 wählen (ungerade)
                    # Nur anwenden, wenn das Grid valide Daten enthält
                    if not np.all(np.isnan(z_data)):
                        z_data = remove_morphological_artifacts(z_data, kernel_size=kernel_size)
                        
                # Wir nutzen das 't', das oben im if/else Block korrekt definiert wurde
                meta = np.array([t[2], t[5], t[0], t[4]])

                attributes = {
                    'crs': src.crs.to_string() if src.crs else "Unknown",
                    'driver': src.driver,
                    'tags': src.tags(),
                    'original_res': current_res,
                    'processing': 'resampled' if (actual_res != current_res) else 'raw_read'
                }

                _save_data_to_npz(save_path, z_data, meta, 'grid', attributes=attributes)
        except Exception as e:
            print(f"Fehler bei {filename}: {e}", flush=True)
                       
def convert_laz_to_npz(input_folder, output_folder, mode='raw', resolution=1.0):
    """
    Importiert LiDAR-Daten (.laz/.las) und konvertiert sie.
    
    Args:
        input_folder (str): Ordner mit .laz/.las Dateien.
        output_folder (str): Zielordner.
        mode (str, optional): 'raw' für Punktwolke, 'grid' für Raster. Defaults to 'raw'.
        resolution (float, optional): Rasterauflösung in Metern. Wenn mode=grid sonst irrelevant. Defaults to 1.0.
    """
    os.makedirs(output_folder, exist_ok=True)
    laz_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.laz', '.las'))]

    for filename in tqdm(laz_files, desc=f"LAZ -> NPZ ({mode})"):
        filepath = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_folder, f"{base_name}_{mode}.npz")

        try:
            laz = laspy.read(filepath)
            x, y, z = laz.x, laz.y, laz.z

            # Attribute sammeln
            normals = None
            colors = None
            attributes = {}

            if hasattr(laz, 'red'):
                red = laz.red
                scale = 65535.0 if red.max() > 255 else 255.0
                colors = np.vstack((laz.red, laz.green, laz.blue)).T / scale
                colors = colors.astype(np.float32)

            try:
                if 'normal_x' in laz.point_format.dimension_names:
                    nx, ny, nz = laz.normal_x, laz.normal_y, laz.normal_z
                    normals = np.vstack((nx, ny, nz)).T.astype(np.float32)
            except: pass

            if hasattr(laz, 'intensity'):
                attributes['intensity'] = np.array(laz.intensity, dtype=np.float32)
            if hasattr(laz, 'classification'):
                attributes['classification'] = np.array(laz.classification, dtype=np.uint8)

            # Stack for cleaning
            points = np.vstack((x, y, z)).T
            
            points, colors, normals, attributes = _remove_invalid_data_points(points, colors, normals, attributes)
            
            if len(points) == 0:
                print(f"Warnung: {filename} enthält nach Bereinigung keine Daten mehr.", flush=True)
                continue

            # Update x, y, z refs after cleaning
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            if mode == 'raw':
                _save_data_to_npz(
                    save_path, points,
                    data_type='raw',
                    colors=colors,
                    normals=normals,
                    attributes=attributes if attributes else None
                )

            elif mode == 'grid':
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                cols = int(np.ceil((x_max - x_min) / resolution))
                rows = int(np.ceil((y_max - y_min) / resolution))
                x_edges = np.linspace(x_min, x_min + cols * resolution, cols + 1)
                y_edges = np.linspace(y_min, y_min + rows * resolution, rows + 1)
                
                # Hier fließen jetzt nur noch saubere Z-Werte ein -> Mean ist korrekt
                z_grid, _, _, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_edges, y_edges])

                colors_grid = None
                if colors is not None:
                    r_grid = binned_statistic_2d(x, y, colors[:,0], statistic='mean', bins=[x_edges, y_edges])[0]
                    g_grid = binned_statistic_2d(x, y, colors[:,1], statistic='mean', bins=[x_edges, y_edges])[0]
                    b_grid = binned_statistic_2d(x, y, colors[:,2], statistic='mean', bins=[x_edges, y_edges])[0]
                    colors_grid = np.dstack((r_grid, g_grid, b_grid)).astype(np.float32)

                meta = np.array([x_min, y_min, resolution, resolution])
                _save_data_to_npz(save_path, z_grid.astype(np.float64), meta, 'grid', colors=colors_grid)

            del x, y, z, points
        except Exception as e:
            print(f"Fehler bei {filename}: {e}", flush=True)

def extract_and_convert_ascii_archives(input_folder, output_folder, target_type='auto', resolution=None):
    """
    Extrahiert gezippte ASCII/XYZ-Dateien und konvertiert sie in NPZ.

    Args:
        input_folder (str): Ordner mit .zip Dateien.
        output_folder (str): Zielordner.
        target_type (str, optional): 'auto', 'grid', 'raw'. Defaults to 'auto'.
        resolution (float, optional): Rasterweite in Metern. Wenn target=grid oder auto sonst irrelevant. Defaults to None.
    """
    os.makedirs(output_folder, exist_ok=True)
    zip_files = [f for f in os.listdir(input_folder) if f.endswith('.zip')]

    for zip_name in tqdm(zip_files, desc="ASCII Zip -> NPZ"):
        zip_path = os.path.join(input_folder, zip_name)
        try:
            def is_three_column_data(line):
                try:
                    parts = line.split()
                    return len(parts) == 3 and all(float(x) or True for x in parts)
                except ValueError: return False

            with zipfile.ZipFile(zip_path, 'r') as zf:
                txt_files = [f for f in zf.namelist() if f.endswith(('.txt', '.xyz', '.asc'))]
                
                if len(txt_files) >= 1:
                    if len(txt_files) > 1:
                        print(f"Warnung: {zip_name} enthält mehrere Textdateien. Nutze {txt_files[0]}.", flush=True)
                        
                    with zf.open(txt_files[0]) as f:
                        lines = [line.decode('latin-1') for line in f]
                        start_index = 0
                        for i, line in enumerate(lines):
                            if is_three_column_data(line):
                                start_index = i
                                break
                        
                        data = np.loadtxt(lines[start_index:], encoding='latin-1').astype(np.float64)

                    data, _, _, _ = _remove_invalid_data_points(data)
                    
                    if len(data) == 0:
                        print(f"Warnung: {zip_name} enthält nur NODATA Werte.", flush=True)
                        continue

                    base_name = os.path.splitext(zip_name)[0]
                    
                    # --- ENTSCHEIDUNG: GRID VS RAW ---
                    save_as_grid = False
                    detected_res = resolution

                    if target_type == 'raw':
                        save_as_grid = False
                    elif target_type == 'grid':
                        save_as_grid = True
                    elif target_type == 'auto':
                        unique_x = np.unique(data[:, 0])
                        if len(unique_x) > 1:
                            diffs = np.diff(unique_x)
                            if np.std(diffs) < 0.01: 
                                save_as_grid = True
                                if detected_res is None:
                                    detected_res = np.median(diffs)
                    
                    # --- SPEICHERN ---
                    if save_as_grid:
                        if detected_res is None:
                             unique_x = np.unique(data[:, 0])
                             if len(unique_x) > 1:
                                 detected_res = np.median(np.diff(unique_x))
                             else:
                                 detected_res = 1.0 
                        
                        detected_res = float(detected_res)
                        x, y, z = data[:, 0], data[:, 1], data[:, 2]
                        x_min, x_max = x.min(), x.max()
                        y_min, y_max = y.min(), y.max()
                        
                        cols = int(np.round((x_max - x_min) / detected_res)) + 1
                        rows = int(np.round((y_max - y_min) / detected_res)) + 1
                        
                        x_edges = np.linspace(x_min - detected_res/2, x_max + detected_res/2, cols + 1)
                        y_edges = np.linspace(y_min - detected_res/2, y_max + detected_res/2, rows + 1)
                        
                        # Binning ist jetzt sicher, da Z sauber ist
                        z_grid, _, _, _ = binned_statistic_2d(
                            x, y, z, 
                            statistic='mean', 
                            bins=[x_edges, y_edges]
                        )
                        z_grid = z_grid.T
                        
                        origin_x = x_min - (detected_res / 2.0)
                        origin_y = y_min - (detected_res / 2.0)
                        
                        meta = np.array([origin_x, origin_y, detected_res, detected_res])
                        
                        save_path = os.path.join(output_folder, f"{base_name}_grid.npz")
                        _save_data_to_npz(save_path, z_grid.astype(np.float64), meta, 'grid')
                        
                    else:
                        save_path = os.path.join(output_folder, f"{base_name}_raw.npz")
                        _save_data_to_npz(save_path, data, data_type='raw')

        except Exception as e:
            print(f"Fehler bei {zip_name}: {e}", flush=True)          
            
def merge_elevation_and_orthophoto(dom_folder, dop_folder, output_folder):
    """Kombiniert ein Höhenmodell (DOM) mit einem Orthophoto (DOP)."""
    os.makedirs(output_folder, exist_ok=True)
    dom_files = sorted([f for f in os.listdir(dom_folder) if f.endswith('.tif')])

    for filename in tqdm(dom_files, desc="Merge DOM+DOP"):
        dom_path = os.path.join(dom_folder, filename)
        dop_path = os.path.join(dop_folder, filename)
        if not os.path.exists(dop_path): continue

        with rasterio.open(dom_path) as src_dom:
            z_data = src_dom.read(1).astype(np.float64)
            
            # Cleaning DOM Nodata
            if src_dom.nodata is not None:
                 z_data[np.isclose(z_data, src_dom.nodata)] = np.nan
            z_data[np.isclose(z_data, NODATA_VALUE)] = np.nan

            meta_raw = src_dom.transform
            meta = np.array([meta_raw[2], meta_raw[5], meta_raw[0], meta_raw[4]])
            attributes = {
                'crs': src_dom.crs.to_string() if src_dom.crs else "Unknown",
                'tags': src_dom.tags()
            }

        with rasterio.open(dop_path) as src_dop:
            rgb = src_dop.read([1, 2, 3]).transpose(1, 2, 0)
            if rgb.dtype == np.uint8: rgb = rgb.astype(np.float64) / 255.0
            else: rgb = rgb.astype(np.float64)

        base_name = os.path.splitext(filename)[0]
        save_path = os.path.join(output_folder, f"{base_name}_merged.npz")
        _save_data_to_npz(save_path, z_data, meta, 'grid', colors=rgb, attributes=attributes)

def export_data_to_npz(save_path, data, meta=None, data_type='grid', colors=None):
    """Wrapper zum manuellen Speichern von Daten."""
    _save_data_to_npz(save_path, data, meta, data_type, colors)
    print(f"Daten gespeichert: {save_path}", flush=True)
