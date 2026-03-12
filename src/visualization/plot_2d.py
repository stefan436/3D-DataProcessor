# src/visualization/plot_2d.py

import os

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.settings import NODATA_VALUE
from core.utils import _collect_npz_file_paths


def render_2d_elevation_plot(source, max_resolution=2000):
    """Erstellt einen kombinierten 2D-Plot für Grid- und Raw-Daten mit globaler Farbskala.

    Implementiert eine intelligente Auflösungs-Reduktion, um Performance zu sichern:
    Wenn die Daten größer als `max_resolution` sind, werden sie automatisch
    runterskaliert (Stride), bevor sie an Matplotlib übergeben werden.
    Dies ist das Äquivalent zur Grid-Size-Abfrage in Vispy, aber automatisiert für Plots.

    Features:
    - 2-Pass Verfahren: Erst globale Min/Max berechnen, dann plotten.
    - Korrekte Darstellung von Nodata-Werten (transparent).
    - Unterstützung für gemischte Raw- und Grid-Daten im selben Plot.

    Args:
        source (str | list): Pfad zu Ordner/Datei oder Liste von npz-Dateien.
        max_resolution (int, optional): Maximale Pixel/Punkte pro Achse für die Anzeige.
                                        Verhindert Memory-Overflows bei riesigen Daten.
                                        Defaults to 2000.
    """        
    files = []
    if isinstance(source, (str, os.PathLike)):
        files = _collect_npz_file_paths(source)
    elif isinstance(source, list):
        files = source

    if not files:
        print("Keine Dateien gefunden.")
        return

    # --- PASS 1: GLOBALE BOUNDS & MIN/MAX BERECHNEN ---
    print("Analysiere globalen Datenbereich...", flush=True)
    g_z_min, g_z_max = np.inf, -np.inf
    g_x_min, g_x_max = np.inf, -np.inf
    g_y_min, g_y_max = np.inf, -np.inf

    loaded_data = [] # Cache

    for f in tqdm(files, desc="Pre-Scan"):
        try:
            with np.load(f, allow_pickle=True) as loaded:
                dtype = str(loaded['type'])
                data = loaded['data']
                meta = loaded['meta'] if 'meta' in loaded else None

                # Z-Werte extrahieren (Nodata filtern!)
                z_vals = None
                # Lokale Bounds initialisieren
                l_xmin, l_xmax, l_ymin, l_ymax = 0, 0, 0, 0

                if dtype == 'grid':
                    z_vals = data.flatten()
                    # Metadaten Grid: x_off, y_off, res_x, res_y
                    x_off, y_off, res_x, res_y = meta
                    rows, cols = data.shape
                    
                    # --- FIX: Robuste Min/Max Berechnung (sortiert) ---
                    # Verhindert Fehler bei negativer Auflösung
                    x_edges = sorted([x_off, x_off + cols * res_x])
                    y_edges = sorted([y_off, y_off + rows * res_y])
                    
                    l_xmin, l_xmax = x_edges[0], x_edges[1]
                    l_ymin, l_ymax = y_edges[0], y_edges[1]

                elif dtype == 'raw':
                    # Raw [N, 3] -> z ist Index 2
                    z_vals = data[:, 2]
                    if len(data) > 0:
                        l_xmin, l_xmax = data[:, 0].min(), data[:, 0].max()
                        l_ymin, l_ymax = data[:, 1].min(), data[:, 1].max()

                # Filter Nodata & NaNs für korrekte Colormap
                if z_vals is not None:
                    valid_mask = (z_vals != NODATA_VALUE) & (~np.isnan(z_vals))
                    if np.any(valid_mask):
                        v_min = np.min(z_vals[valid_mask])
                        v_max = np.max(z_vals[valid_mask])
                        if v_min < g_z_min: g_z_min = v_min
                        if v_max > g_z_max: g_z_max = v_max

                # Bounds update
                if l_xmin < g_x_min: g_x_min = l_xmin
                if l_xmax > g_x_max: g_x_max = l_xmax
                if l_ymin < g_y_min: g_y_min = l_ymin
                if l_ymax > g_y_max: g_y_max = l_ymax

                # Daten für Pass 2 speichern
                loaded_data.append({'type': dtype, 'data': data, 'meta': meta, 'file': os.path.basename(f)})

        except Exception as e:
            print(f"Skip {f}: {e}")

    if g_z_min == np.inf:
        print("Keine validen Z-Daten gefunden.")
        return

    # --- FIX: Singuläre Achsen verhindern (Padding) ---
    if g_x_max <= g_x_min: 
        g_x_min -= 1.0; g_x_max += 1.0
    if g_y_max <= g_y_min: 
        g_y_min -= 1.0; g_y_max += 1.0

    print(f"Global Z: {g_z_min:.2f} bis {g_z_max:.2f} | Area: {g_x_min:.1f},{g_y_min:.1f} bis {g_x_max:.1f},{g_y_max:.1f}")

    # --- PASS 2: PLOTTING MIT AUFLÖSUNGS-LIMITER ---
    fig, ax = plt.subplots(figsize=(9, 7))
    norm = plt.Normalize(vmin=g_z_min, vmax=g_z_max)
    cmap = plt.get_cmap('viridis')

    img_handle = None

    for item in tqdm(loaded_data, desc="Plotting"):
        if item['type'] == 'grid':
            z = item['data']
            rows, cols = z.shape
            x_off, y_off, res_x, res_y = item['meta']

            # --- Smart Downsampling (Stride) ---
            step_row = max(1, int(rows / max_resolution))
            step_col = max(1, int(cols / max_resolution))

            z_view = z[::step_row, ::step_col]
            z_masked = np.ma.masked_where((z_view == NODATA_VALUE) | (np.isnan(z_view)), z_view)

            # Extent Berechnung für imshow (Left, Right, Bottom, Top)
            # Matplotlib erwartet Bottom < Top, außer wir wollen Achsen drehen.
            # Wir berechnen die physischen Ecken basierend auf dem View:
            
            # Die originale Ausdehnung:
            e_left = x_off
            e_right = x_off + cols * res_x
            e_bottom = y_off + rows * res_y # Bei GeoTiff (neg res) ist das "unten"
            e_top = y_off                   # Bei GeoTiff ist das "oben"
            
            # Falls res_y negativ ist, ist e_bottom < e_top. Imshow kommt damit klar (origin='upper' oder explizit).
            # Wir nutzen 'lower' aber geben Extent passend: [min_x, max_x, min_y, max_y]
            # ACHTUNG: imshow erwartet [left, right, bottom, top].
            
            extent = [e_left, e_right, e_bottom, e_top] 

            img_handle = ax.imshow(z_masked, extent=extent, origin='upper', 
                                cmap=cmap, norm=norm, interpolation='none')

        elif item['type'] == 'raw':
            xyz = item['data']
            num_points = len(xyz)
            target_count = max_resolution * max_resolution
            step = max(1, int(num_points / target_count))
            xyz_sub = xyz[::step]

            img_handle = ax.scatter(xyz_sub[:, 0], xyz_sub[:, 1], c=xyz_sub[:, 2],
                                    s=1, cmap=cmap, norm=norm, edgecolors='none', alpha=0.8)

    # --- VIEWPORT SETZEN ---
    ax.set_xlim(g_x_min, g_x_max)
    ax.set_ylim(g_y_min, g_y_max)
    ax.set_aspect('equal')

    plt.title(f"Combined 2D View (Z: {g_z_min:.1f}m - {g_z_max:.1f}m)")
    if img_handle:
        plt.colorbar(img_handle, ax=ax, label='Height (m)')
    print("Starte plt.show()...", flush=True)
    plt.show()
