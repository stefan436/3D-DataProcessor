# src/core/utils.py

import os

import numpy as np
from scipy.stats import binned_statistic_2d

from config.settings import NODATA_VALUE


def _remove_invalid_data_points(points, colors=None, normals=None, attributes=None):
    """Filtert ungültige Punkte (NaN, Inf, NODATA) aus Rohdaten.
    
    Wichtig: Wendet die Maske synchron auf alle Zusatzattribute an.

    Args:
        points (np.ndarray): Punktwolke (N, 3).
        colors (np.ndarray, optional): Farben (N, 3).
        normals (np.ndarray, optional): Normalen (N, 3).
        attributes (dict, optional): Dictionary mit Arrays der Länge N.

    Returns:
        tuple: (points_clean, colors_clean, normals_clean, attributes_clean)
    """
    if points is None or len(points) == 0:
        return points, colors, normals, attributes

    # 1. Z-Werte extrahieren
    z = points[:, 2]

    # 2. Maske erstellen: Kein NaN, kein Inf, nicht NODATA (mit Toleranz für Float)
    # np.isclose ist wichtig, da float Vergleiche mit == oft scheitern
    is_nodata = np.isclose(z, NODATA_VALUE, atol=1e-5)
    mask = np.isfinite(z) & (~is_nodata)

    # Wenn alles sauber ist, spare RAM und Zeit
    if np.all(mask):
        return points, colors, normals, attributes

    # 3. Maske anwenden
    p_clean = points[mask]
    c_clean = colors[mask] if colors is not None else None
    n_clean = normals[mask] if normals is not None else None
    
    attr_clean = {}
    if attributes is not None:
        for k, v in attributes.items():
            if isinstance(v, (np.ndarray, list)) and len(v) == len(points):
                attr_clean[k] = np.array(v)[mask]
            else:
                attr_clean[k] = v # Skalare Attribute behalten

    return p_clean, c_clean, n_clean, attr_clean

def _save_data_to_npz(save_path, data, meta=None, data_type='grid', colors=None, normals=None, attributes=None):
    """Speichert Daten im internen komprimierten .npz Format."""
    save_dict = {'data': data, 'type': data_type}
    if meta is not None: save_dict['meta'] = meta
    if colors is not None: save_dict['colors'] = colors
    if normals is not None: save_dict['normals'] = normals
    if attributes is not None: save_dict['attributes'] = attributes

    np.savez_compressed(save_path, **save_dict)

def _collect_npz_file_paths(path):
    if os.path.isfile(path) and path.endswith('.npz'): return [path]
    elif os.path.isdir(path): return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')]
    return []

def extract_point_cloud_from_npz(npz_path):
    """Lädt eine NPZ-Datei und gibt sie als Punktwolke zurück.

    Konvertiert Grid-Daten automatisch in XYZ-Punkte, falls nötig.

    Args:
        npz_path (str): Pfad zur Datei.

    Returns:
        tuple: (xyz_points [N,3], rgb_colors [N,3] or None)
    """
    loaded = np.load(npz_path, allow_pickle=True)
    if str(loaded['type']) == 'raw':
        return loaded['data'], (loaded['colors'] if 'colors' in loaded else None)
    elif str(loaded['type']) == 'grid':
        z = loaded['data']; meta = loaded['meta']
        rgb = loaded['colors'] if 'colors' in loaded else None
        rows, cols = z.shape
        x = meta[0] + np.arange(cols) * meta[2] + (meta[2]/2)
        y = meta[1] + np.arange(rows) * meta[3] + (meta[3]/2)
        xg, yg = np.meshgrid(x, y)
        xyz = np.column_stack((xg.flatten(), yg.flatten(), z.flatten()))
        rgb = rgb.reshape(-1, 3) if rgb is not None else None
        mask = ~np.isnan(xyz[:, 2])
        return xyz[mask], (rgb[mask] if rgb is not None else None)

def print_npz_metadata_structure(filepath):
    """Debug-Funktion: Gibt die interne Struktur einer NPZ-Datei auf der Konsole aus.

    Args:
        filepath (str): Pfad zur .npz Datei.
    """
    with np.load(filepath, allow_pickle=True) as data:
        print(f"=== METADATA: {os.path.basename(filepath)} ===")
        print(f"Files contained in npz: {data.files}")
        for file in data.files:
            content = data[file]
            if file == 'data':
                print(f"Content of data (shape): {content.shape}")
            else:
                print(f"Content of {file}: {content}")
