# src/visualization/view_3d.py

import numpy as np
import vispy
import vispy.app
vispy.use(app='pyqt6') 
import vispy.scene
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform
from tqdm import tqdm
from scipy.stats import binned_statistic_2d

from core.utils import _collect_npz_file_paths, extract_point_cloud_from_npz



def render_interactive_3d_scatter_plot(input_path, x_res=1000, y_res=1000, use_full=False, satellite_color=False, XYZ_axis=True):
    """Startet den Vispy Scatter Viewer (Argument-basiert für GUI)."""

    
    files = _collect_npz_file_paths(input_path)
    if not files: return

    # --- SCHRITT 1: PRE-SCAN ---
    print("Scanne globale Z-Werte...", flush=True)
    g_z_min, g_z_max = np.inf, -np.inf

    for f in tqdm(files, desc="Pre-Scan Z"):
        try:
            with np.load(f, allow_pickle=True) as loaded:
                if str(loaded['type']) == 'grid':
                    z_data = loaded['data']
                    local_min, local_max = np.nanmin(z_data), np.nanmax(z_data)
                else: 
                    z_data = loaded['data'][:, 2]
                    local_min, local_max = np.nanmin(z_data), np.nanmax(z_data)

                if local_min < g_z_min: g_z_min = local_min
                if local_max > g_z_max: g_z_max = local_max
        except Exception: pass

    z_range = g_z_max - g_z_min
    if z_range <= 0: z_range = 1.0

    print(f"Modus: {'FULL' if use_full else 'REDUCED'} | Res: {x_res}x{y_res}", flush=True)

    # --- SCHRITT 3: VISUALISIERUNG ---
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='grey')
    view = canvas.central_widget.add_view(camera='arcball')

    total_x_min, total_x_max = np.inf, -np.inf
    total_y_min, total_y_max = np.inf, -np.inf
    total_z_min, total_z_max = np.inf, -np.inf

    global_offset = None
    cmap = vispy.color.get_colormap('plasma')

    for f in tqdm(files, desc="Plotting"):
        try:
            xyz, rgb = extract_point_cloud_from_npz(f)
            if len(xyz) == 0: continue
            z_absolute = xyz[:, 2]

            if global_offset is None:
                global_offset = xyz[0].copy()

            xyz = (xyz - global_offset).astype(np.float32)
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

            total_x_min = min(total_x_min, np.nanmin(x)); total_x_max = max(total_x_max, np.nanmax(x))
            total_y_min = min(total_y_min, np.nanmin(y)); total_y_max = max(total_y_max, np.nanmax(y))
            total_z_min = min(total_z_min, np.nanmin(z)); total_z_max = max(total_z_max, np.nanmax(z))

            if use_full:
                pos = xyz
                mask = ~np.isnan(pos[:, 2])
                pos = pos[mask]
                
                if satellite_color and rgb is not None:
                    colors = rgb[mask]
                else:
                    norm_z = (z_absolute[mask] - g_z_min) / z_range
                    colors = cmap.map(norm_z)
                pt_size = 1
            else:
                # Binning
                x_ed = np.linspace(x.min(), x.max(), int(x_res) + 1)
                y_ed = np.linspace(y.min(), y.max(), int(y_res) + 1)

                z_s, _, _, _ = binned_statistic_2d(x, y, z, statistic='mean', bins=[x_ed, y_ed])
                z_s_abs = z_s + global_offset[2]

                xc = (x_ed[:-1] + x_ed[1:]) / 2
                yc = (y_ed[:-1] + y_ed[1:]) / 2
                xg, yg = np.meshgrid(xc, yc, indexing='ij')
                pos = np.column_stack([xg.flatten(), yg.flatten(), z_s.flatten()])

                mask = ~np.isnan(pos[:, 2])
                pos = pos[mask]
                z_s_abs_masked = z_s_abs.flatten()[mask]

                if satellite_color and rgb is not None:
                    r_s = binned_statistic_2d(x, y, rgb[:,0], statistic='mean', bins=[x_ed, y_ed])[0]
                    g_s = binned_statistic_2d(x, y, rgb[:,1], statistic='mean', bins=[x_ed, y_ed])[0]
                    b_s = binned_statistic_2d(x, y, rgb[:,2], statistic='mean', bins=[x_ed, y_ed])[0]
                    colors = np.column_stack([r_s.flatten(), g_s.flatten(), b_s.flatten()])[mask]
                else:
                    norm_z = (z_s_abs_masked - g_z_min) / z_range
                    colors = cmap.map(norm_z)
                pt_size = 2

            if len(pos) > 0:
                view.add(visuals.Markers(pos=pos, size=pt_size, edge_width=0, antialias=0, face_color=colors, scaling='scene'))
        except Exception as e:
            print(f"Fehler bei {f}: {e}")

    if total_x_min != np.inf:
        view.camera.set_range(x=[total_x_min, total_x_max], y=[total_y_min, total_y_max], z=[total_z_min, total_z_max])
        if XYZ_axis:
             visuals.XYZAxis(parent=view.scene).transform = STTransform(
                translate=(total_x_min, total_y_min, total_z_min),
                scale=(total_x_max-total_x_min, total_y_max-total_y_min, total_z_max-total_z_min)
            )

    vispy.app.run()

def render_interactive_3d_surface_plot(input_path, x_res=1000, y_res=1000, use_native=False, XYZ_axis=True):
    """Startet den Vispy Surface Viewer (Argument-basiert für GUI)."""
    files = _collect_npz_file_paths(input_path)
    if not files: return

    # Scan für Grid-Info
    max_cols, max_rows = 1000, 1000
    is_grid_source = False
    try:
        with np.load(files[0], allow_pickle=True) as l:
            if str(l['type']) == 'grid':
                is_grid_source = True
                max_rows, max_cols = l['data'].shape
    except: pass

    # Override Resolutions if Native
    if use_native:
        print("Native Auflösung angefordert.", flush=True)
    else:
        print(f"Ziel-Auflösung: {x_res}x{y_res}", flush=True)

    print(f"Lade Daten (konvertiere zu float32)...", flush=True)

    all_xyz = []
    x_min, x_max = np.float32(np.inf), np.float32(-np.inf)
    y_min, y_max = np.float32(np.inf), np.float32(-np.inf)
    global_offset = None

    for f in tqdm(files, desc="Loading (f32)"):
        try:
            xyz_raw, rgb_raw = extract_point_cloud_from_npz(f)
            if len(xyz_raw) == 0: continue

            if global_offset is None:
                global_offset = xyz_raw[0].copy()

            xyz_shifted = (xyz_raw - global_offset).astype(np.float32)
            x_min = min(x_min, np.nanmin(xyz_shifted[:, 0]))
            x_max = max(x_max, np.nanmax(xyz_shifted[:, 0]))
            y_min = min(y_min, np.nanmin(xyz_shifted[:, 1]))
            y_max = max(y_max, np.nanmax(xyz_shifted[:, 1]))

            all_xyz.append(xyz_shifted)
        except Exception: pass

    if not all_xyz: return
    full_xyz = np.vstack(all_xyz)

    # Globales Grid berechnen
    cols, rows = int(x_res), int(y_res)
    
    if use_native:
        if is_grid_source:
            pixel_width = (x_max - x_min) / max_cols if max_cols > 0 else 1.0
            cols = int((x_max - x_min) / pixel_width)
            rows = int((y_max - y_min) / pixel_width)
        else:
            step = 0.5
            cols = int((x_max - x_min) / step)
            rows = int((y_max - y_min) / step)
            if cols * rows > 25000000:
                print("WARNUNG: >25 Mio Pixel. Limitiere auf 2000px.")
                cols, rows = 2000, 2000

    cols, rows = max(2, cols), max(2, rows)
    x_edges = np.linspace(x_min, x_max, cols + 1, dtype=np.float32)
    y_edges = np.linspace(y_min, y_max, rows + 1, dtype=np.float32)

    z_grid, _, _, _ = binned_statistic_2d(
        full_xyz[:, 0], full_xyz[:, 1], full_xyz[:, 2],
        statistic='mean', bins=[x_edges, y_edges]
    )
    z_grid = z_grid.astype(np.float32)

    z_min_val = np.nanmin(z_grid)
    z_safe = np.nan_to_num(z_grid, nan=z_min_val)

    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view(camera='turntable')

    surface = visuals.SurfacePlot(
        z=z_safe,
        x=np.linspace(x_min, x_max, cols, dtype=np.float32),
        y=np.linspace(y_min, y_max, rows, dtype=np.float32),
        shading='smooth'
    )
    surface.cmap = 'viridis'
    view.add(surface)
    view.camera.set_range(x=[x_min, x_max], y=[y_min, y_max], z=[z_min_val, np.nanmax(z_safe)])

    if XYZ_axis:
        visuals.XYZAxis(parent=view.scene).transform = STTransform(
            translate=(x_min, y_min, z_min_val),
            scale=(x_max-x_min, y_max-y_min, np.nanmax(z_safe)-z_min_val)
        )

    vispy.app.run()
