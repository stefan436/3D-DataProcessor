# src/gui/main_window.py

import os
import time
import multiprocessing
import psutil

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QToolButton, QLabel, QPushButton, 
                             QFileDialog, QPlainTextEdit, QSplitter,
                             QMessageBox, QFrame, QGroupBox, QComboBox, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QIcon, QTextCursor, QFont


# GUI Komponenten
from .dialogs import ParameterDialog
from .workers import WorkerThread

# IO Operationen
from core.io_operations import (
    convert_geotiff_to_npz, 
    convert_laz_to_npz, 
    extract_and_convert_ascii_archives, 
    merge_elevation_and_orthophoto
)

# Processing
from core.processing import (
    crop_dataset_to_bounds, 
    batch_transform_dataset_crs, 
    scale_elevation_in_dataset, 
    rasterize_point_cloud_to_grid, 
    merge_npz_datasets
)

# Meshing
from core.meshing import (
    batch_process_npz_to_meshes, 
    batch_generate_mesh_enclosures, 
    concatenate_ply_files_memory_efficiently
)

# Utils
from core.utils import print_npz_metadata_structure

# Visualisierung
from visualization.plot_2d import render_2d_elevation_plot
from visualization.view_3d import render_interactive_3d_scatter_plot, render_interactive_3d_surface_plot


class HeightProfileApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeightProfileApp")
        self.resize(1200, 850)
        
        font = QFont("Segoe UI", 10)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        QApplication.setFont(font)

        self.current_input_dir = ""
        self.current_output_dir = ""
        
        self.thread = None
        self.current_process = None 
        self.task_start_time = None
        self.current_task_text = ""
        
        self.init_ui()
        
        self.sys_timer = QTimer()
        self.sys_timer.timeout.connect(self.update_status_bar_info)
        self.sys_timer.start(1000)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 1. Header (Pfade)
        self.header = self.create_project_header()
        main_layout.addWidget(self.header)

        # 2. Ribbon Tabs
        self.ribbon = QTabWidget()
        self.ribbon.setFixedHeight(150) 
        
        self.create_import_tab()
        self.create_processing_tab()
        self.create_meshing_tab()
        self.create_tools_tab()
        self.create_vis_tab()
        
        main_layout.addWidget(self.ribbon)

        # 3. Konsole
        splitter = QSplitter(Qt.Orientation.Vertical)
        lbl_console = QLabel("PROZESS PROTOKOLL")
        lbl_console.setStyleSheet("color: #666; font-size: 8pt; font-weight: bold; margin-left: 2px;")
        main_layout.addWidget(lbl_console)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        console_font = QFont("Consolas", 10)
        console_font.setStyleHint(QFont.StyleHint.Monospace)
        self.console.setFont(console_font)
        splitter.addWidget(self.console)
        main_layout.addWidget(splitter)

        # 4. Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.setSizeGripEnabled(False) 
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_DialogCancelButton))
        self.btn_stop.setStyleSheet("""
            QPushButton { background-color: #c42b1c; border: none; color: white; font-weight: bold; padding: 2px 10px; border-radius: 0px; }
            QPushButton:hover { background-color: #e81123; }
        """)
        self.btn_stop.clicked.connect(self.stop_execution)
        self.btn_stop.setVisible(False)
        self.status_bar.addWidget(self.btn_stop)
        
        spacer = QWidget()
        spacer.setFixedWidth(10)
        self.status_bar.addWidget(spacer)

        self.lbl_task_name = QLabel("")
        self.lbl_task_name.setStyleSheet("color: #4a6572; font-weight: bold;")
        self.status_bar.addWidget(self.lbl_task_name)

        self.lbl_cpu = QLabel("CPU: 0%")
        self.lbl_cpu.setFixedWidth(80)
        self.lbl_cpu.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bar.addPermanentWidget(self.lbl_cpu)

        self.lbl_ram = QLabel("RAM: N/A")
        self.lbl_ram.setFixedWidth(180)
        self.lbl_ram.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bar.addPermanentWidget(self.lbl_ram)

    def create_project_header(self):
        group = QGroupBox("PROJEKT PFADE")
        group.setObjectName("HeaderBox")
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(10)

        grid = QWidget()
        l_grid = QHBoxLayout(grid)
        l_grid.setContentsMargins(0,0,0,0)
        
        v_in = QVBoxLayout()
        lbl_in = QLabel("Input Verzeichnis")
        lbl_in.setStyleSheet("color: #888; font-size: 9pt;")
        self.txt_input = QLineEdit()
        self.txt_input.setReadOnly(True)
        self.txt_input.setPlaceholderText("Input Ordner wählen...")
        
        h_in_btns = QHBoxLayout()
        h_in_btns.setSpacing(5)
        h_in_btns.addWidget(self.txt_input)
        btn_in = QPushButton("...")
        btn_in.clicked.connect(self.select_input)
        h_in_btns.addWidget(btn_in)
        v_in.addWidget(lbl_in)
        v_in.addLayout(h_in_btns)
        
        btn_swap = QPushButton("⇄")
        btn_swap.setToolTip("Pfade tauschen")
        btn_swap.setFixedSize(40, 40)
        btn_swap.setStyleSheet("""
            QPushButton { font-size: 18pt; color: #4a6572; border: 1px solid #444; background: #2d2d30; border-radius: 20px; }
            QPushButton:hover { background: #3e3e42; border-color: #4a6572; }
        """)
        btn_swap.clicked.connect(self.swap_paths)
        
        v_out = QVBoxLayout()
        lbl_out = QLabel("Output Verzeichnis")
        lbl_out.setStyleSheet("color: #888; font-size: 9pt;")
        self.txt_output = QLineEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setPlaceholderText("Output Ordner wählen...")

        h_out_btns = QHBoxLayout()
        h_out_btns.setSpacing(5)
        h_out_btns.addWidget(self.txt_output)
        btn_out = QPushButton("...")
        btn_out.clicked.connect(self.select_output)
        h_out_btns.addWidget(btn_out)
        v_out.addWidget(lbl_out)
        v_out.addLayout(h_out_btns)

        l_grid.addLayout(v_in)
        l_grid.addWidget(btn_swap)
        l_grid.addLayout(v_out)
        layout.addWidget(grid)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #3e3e42;")
        layout.addWidget(line)

        h_settings = QHBoxLayout()
        h_settings.addStretch()
        
        lbl_dec = QLabel("Dezimal-Trennzeichen:")
        lbl_dec.setStyleSheet("color: #ccc; font-size: 9pt;")
        h_settings.addWidget(lbl_dec)

        self.combo_dec_sep = QComboBox()
        self.combo_dec_sep.setFixedWidth(150)
        self.combo_dec_sep.addItem("Punkt ( . ) - Int.", ".") 
        self.combo_dec_sep.addItem("Komma ( , ) - DE", ",")
        self.combo_dec_sep.setStyleSheet("QComboBox { background-color: #333337; color: #f1f1f1; border: 1px solid #555; padding: 4px; }")
        h_settings.addWidget(self.combo_dec_sep)

        layout.addLayout(h_settings)
        return group

    def add_ribbon_btn(self, layout, text, icon_name, func):
        btn = QToolButton()
        btn.setText(text)
        icon = self.style().standardIcon(getattr(self.style().StandardPixmap, icon_name, self.style().StandardPixmap.SP_FileIcon))
        btn.setIcon(icon)
        btn.setIconSize(QSize(36, 36))
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn.setFixedWidth(100)
        btn.setFixedHeight(85)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(func)
        layout.addWidget(btn)

    # --- TABS DEFINITION ---
    def create_import_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); layout.setAlignment(Qt.AlignmentFlag.AlignLeft); layout.setContentsMargins(15, 10, 15, 10)
        self.add_ribbon_btn(layout, "GeoTIFF", "SP_FileIcon", self.prompt_geotiff)
        self.add_ribbon_btn(layout, "LAZ / LAS", "SP_DirIcon", self.prompt_laz)
        self.add_ribbon_btn(layout, "ASCII Zip", "SP_FileIcon", self.prompt_ascii)
        self.add_ribbon_btn(layout, "DOM+DOP", "SP_FileDialogDetailedView", self.prompt_dom_dop)
        self.ribbon.addTab(tab, "1. IMPORT")

    def create_processing_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); layout.setAlignment(Qt.AlignmentFlag.AlignLeft); layout.setContentsMargins(15, 10, 15, 10)
        self.add_ribbon_btn(layout, "Crop (Box)", "SP_MediaPlay", self.prompt_crop)
        self.add_ribbon_btn(layout, "Koord. Trans", "SP_DriveNetIcon", self.prompt_coord_transform)
        self.add_ribbon_btn(layout, "Z-Stretch", "SP_ArrowUp", self.prompt_z_stretch)
        self.add_ribbon_btn(layout, "Raw->Grid", "SP_BrowserReload", self.prompt_raw2grid)
        self.add_ribbon_btn(layout, "Merge Tile", "SP_FileDialogListView", self.prompt_merge_npz)
        self.ribbon.addTab(tab, "2. PROCESSING")

    def create_meshing_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); layout.setAlignment(Qt.AlignmentFlag.AlignLeft); layout.setContentsMargins(15, 10, 15, 10)
        self.add_ribbon_btn(layout, "Start Meshing", "SP_ComputerIcon", self.prompt_meshing)
        self.add_ribbon_btn(layout, "Wände/Boden", "SP_BrowserReload", self.prompt_walls)
        self.ribbon.addTab(tab, "3. MESHING")

    def create_tools_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); layout.setAlignment(Qt.AlignmentFlag.AlignLeft); layout.setContentsMargins(15, 10, 15, 10)
        self.add_ribbon_btn(layout, "PLY Join", "SP_DriveCDIcon", self.prompt_combine_ply)
        self.add_ribbon_btn(layout, "Inspect NPZ", "SP_MessageBoxInformation", self.prompt_inspect_npz)
        self.ribbon.addTab(tab, "TOOLS")

    def create_vis_tab(self):
        tab = QWidget(); layout = QHBoxLayout(tab); layout.setAlignment(Qt.AlignmentFlag.AlignLeft); layout.setContentsMargins(15, 10, 15, 10)
        self.add_ribbon_btn(layout, "Plot 2D", "SP_FileDialogListView", self.launch_plot_2d)
        self.add_ribbon_btn(layout, "Scatter 3D", "SP_DesktopIcon", self.launch_scatter)
        self.add_ribbon_btn(layout, "Surface 3D", "SP_DesktopIcon", self.launch_surface)
        self.ribbon.addTab(tab, "4. ANSICHT")

    # --- LOGIK UND DIALOGE ---
    def select_input(self):
        d = QFileDialog.getExistingDirectory(self, "Input Ordner")
        if d: self.current_input_dir = d; self.txt_input.setText(d); self.log_message(f"Input gesetzt: {d}\n")
    
    def select_output(self):
        d = QFileDialog.getExistingDirectory(self, "Output Ordner")
        if d: self.current_output_dir = d; self.txt_output.setText(d); self.log_message(f"Output gesetzt: {d}\n")

    def swap_paths(self):
        self.current_input_dir, self.current_output_dir = self.current_output_dir, self.current_input_dir
        self.txt_input.setText(self.current_input_dir)
        self.txt_output.setText(self.current_output_dir)
        self.log_message(f"Pfade getauscht.\n")

    def log_message(self, msg, replace=False):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        fmt = cursor.charFormat()
        if "ERROR" in msg or "FEHLER" in msg:
            fmt.setForeground(Qt.GlobalColor.red)
        elif "WARNUNG" in msg:
            fmt.setForeground(Qt.GlobalColor.yellow)
        else:
            fmt.clearForeground() 
        
        if replace:
            cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.setCharFormat(fmt)
            cursor.insertText(msg)
        else:
            if self.console.document().characterCount() > 0:
                if cursor.positionInBlock() > 0: cursor.insertText("\n")
            cursor.setCharFormat(fmt)
            cursor.insertText(msg)

        self.console.setTextCursor(cursor)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def check_ready(self):
        if not self.current_input_dir or not self.current_output_dir:
            QMessageBox.warning(self, "Unvollständig", "Bitte wählen Sie Input- und Output-Ordner.")
            return False
        return True

    def update_status_bar_info(self):
        # CPU
        cpu = psutil.cpu_percent()
        self.lbl_cpu.setText(f"CPU: {cpu}%")
        if cpu > 90: self.lbl_cpu.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        else: self.lbl_cpu.setStyleSheet("color: #cccccc;")

        # RAM
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        percent = mem.percent
        self.lbl_ram.setText(f"RAM: {used_gb:.1f} / {total_gb:.0f} GB ({percent}%)")
        if percent > 90: self.lbl_ram.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        elif percent > 75: self.lbl_ram.setStyleSheet("color: orange;")
        else: self.lbl_ram.setStyleSheet("color: #cccccc;")
        
        # Timer
        if self.task_start_time is not None:
            elapsed = int(time.time() - self.task_start_time)
            m, s = divmod(elapsed, 60)
            h, m = divmod(m, 60)
            self.lbl_task_name.setText(f"LÄUFT: {self.current_task_text} | {h:02}:{m:02}:{s:02}")

    def set_ui_running(self, task_name):
        self.btn_stop.setVisible(True)
        self.current_task_text = task_name.upper()
        self.task_start_time = time.time()
        self.update_status_bar_info()

    def reset_ui_state(self):
        self.btn_stop.setVisible(False)
        self.lbl_task_name.setText("")
        self.task_start_time = None
        self.current_task_text = ""

    def on_task_success(self):
        self.reset_ui_state()
        self.status_bar.showMessage("Fertig.", 5000)
        QMessageBox.information(self, "Erfolg", "Vorgang abgeschlossen")

    def stop_execution(self):
        killed = False
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
            self.log_message("\n!!! THREAD ABGEBROCHEN !!!\n")
            killed = True
        if self.current_process and self.current_process.is_alive():
            self.current_process.terminate()
            self.current_process.join()
            self.log_message("\n!!! PROZESS GESTOPPT !!!\n")
            self.current_process = None
            killed = True
        if killed:
            self.reset_ui_state()
            self.status_bar.showMessage("Abgebrochen.", 5000)

    def run_task(self, func, *args, **kwargs):
        if not self.check_ready(): return
        task_name = func.__name__.replace("_", " ").title()
        self.console.appendPlainText(f"\n--- Start: {task_name} ---\n")
        self.set_ui_running(task_name)
        
        # Der Worker nimmt Input/Output immer als erste Argumente
        self.thread = WorkerThread(func, self.current_input_dir, self.current_output_dir, *args, **kwargs)
        self.thread.log_signal.connect(self.log_message)
        self.thread.finished_signal.connect(self.on_task_success)
        self.thread.start()

    # --- WRAPPERS FÜR PARAMETER (V8.0) ---

    def prompt_geotiff(self):
        dlg = ParameterDialog("GeoTIFF Import", {
            'kranfilter': {
                'type': bool, 'default': False, 'label': 'Kranfilter',
                'help': "Ob ein Filter der Kräne in DOMs herausfiltert Aktiviert werden soll. Über Gaus'schen Kernel. Filtert Teils auch Antennen."
            },
            'kernel_size': {
                'type': int, 'default': 6, 'label': 'Kernel Size',
                'help': "Anzahl der Pixel in der Umgebung die Relevant sind für den Filter. Hohe Werte filtern mehr. Gute range ist 5-7."
            },
            'target_res': {
                'type': float, 'default': 0.0, 'label': 'Ziel-Auflösung [m]',
                'help': "Ziel-Auflösung in Metern. 0.0 oder leer für Originalauflösung (kein Binning)."
            }
        }, self)
        if dlg.exec():
            d = dlg.get_data()
            target_res = d['target_res'] if d['target_res'] > 0 else None
            self.run_task(convert_geotiff_to_npz, kranfilter=d['kranfilter'], kernel_size=d['kernel_size'], target_resolution=target_res)

    def prompt_laz(self):
        dlg = ParameterDialog("LAZ Import", {
            'mode': {
                'type': 'combo', 'options': ['raw', 'grid'], 'label': 'Modus',
                'help': "'raw' für Punktwolke, 'grid' für Raster."
            }, 
            'res': {
                'type': float, 'default': 1.0, 'label': 'Auflösung [m]',
                'help': "Rasterauflösung in Metern. Wenn mode=grid sonst irrelevant."
            }
        }, self)
        if dlg.exec(): 
            d = dlg.get_data()
            self.run_task(convert_laz_to_npz, mode=d['mode'], resolution=d['res'])

    def prompt_ascii(self):
        dlg = ParameterDialog("ASCII Import", {
            'target': {
                'type': 'combo', 'options': ['auto', 'grid', 'raw'], 'label': 'Typ',
                'help': "'auto', 'grid', 'raw'."
            },
            'res': {
                'type': float, 'default': 1.0, 'label': 'Auflösung [m]',
                'help': "Rasterweite in Metern. Wenn target=grid oder auto sonst irrelevant."
            }
        }, self)
        if dlg.exec(): 
            d = dlg.get_data()
            self.run_task(extract_and_convert_ascii_archives, target_type=d['target'], resolution=d['res'])

    def prompt_dom_dop(self):
        if not self.check_ready(): return
        dop = QFileDialog.getExistingDirectory(self, "Wähle DOP Ordner")
        if dop: 
            self.console.appendPlainText("\n--- Merge DOM+DOP ---\n")
            self.set_ui_running("Merge DOM+DOP")
            self.thread = WorkerThread(merge_elevation_and_orthophoto, self.current_input_dir, dop, self.current_output_dir)
            self.thread.log_signal.connect(self.log_message)
            self.thread.finished_signal.connect(self.on_task_success)
            self.thread.start()

    def prompt_z_stretch(self):
        dlg = ParameterDialog("Z-Überhöhung", {
            'factor': {
                'type': float, 'default': 2.0, 'label': 'Faktor', 
                'help': "Multiplikator für die Höhe (z.B. 2.0)."
            }
        }, self)
        if dlg.exec(): 
            self.run_task(scale_elevation_in_dataset, factor=dlg.get_data()['factor'])

    def prompt_raw2grid(self):
        dlg = ParameterDialog("Raw zu Grid", {
            'res': {
                'type': float, 'default': 1.0, 'label': 'Rasterweite [m]', 
                'help': "Gewünschte Rasterweite in Metern (z.B. 1.0). Gute wahl ist die ursprüngliche Auflösung + 2 zu wählen (Wenn koordinaten transformiert wurden)."
            }
        }, self)
        if dlg.exec(): 
            self.run_task(rasterize_point_cloud_to_grid, resolution=dlg.get_data()['res'])

    def prompt_crop(self):
        dlg = ParameterDialog("Crop (Bounding Box)", {
            'xmin': {'type': float, 'label': 'X Min'}, 
            'xmax': {'type': float, 'label': 'X Max'}, 
            'ymin': {'type': float, 'label': 'Y Min'}, 
            'ymax': {'type': float, 'label': 'Y Max'},
            'data_crs': {
                'type': str, 'default': 'EPSG:25832', 'label': 'Daten CRS', 
                'help': 'CRS der Daten (z.B. "EPSG:25832").'
            },
            'bounds_crs': {
                'type': str, 'default': 'EPSG:25832', 'label': 'Bounds CRS', 
                'help': 'CRS der Bounds (z.B. "EPSG:4326" für GPS).'
            }
        }, self)
        if dlg.exec():
            d = dlg.get_data()
            self.run_task(crop_dataset_to_bounds, 
                        bounds=(d['xmin'], d['xmax'], d['ymin'], d['ymax']), 
                        data_crs=d['data_crs'], bounds_crs=d['bounds_crs'])

    def prompt_coord_transform(self):
        dlg = ParameterDialog("Koordinaten Transform", {
            'src': {
                'type': str, 'default': 'EPSG:25832', 'label': 'Quelle',
                'help': 'Quell-Koordinatensystem (z.B. "EPSG:25832").'
            },
            'tgt': {
                'type': str, 'default': 'EPSG:3035', 'label': 'Ziel',
                'help': 'Ziel-Koordinatensystem (z.B. "EPSG:4326" oder "EPSG:3035").'
            },
            'mode': {
                'type': 'combo', 'options': ['exact', 'fast'], 'label': 'Grid Modus', 
                'help': "Strategie für Grid-Daten.\n- 'exact': (Empfohlen) Konvertiert Grid -> Raw (Punktwolke).\n  Garantiert positionsgetreue Transformation jedes Pixels (Rotation!), ändert aber Datentyp.\n- 'fast': Transformiert nur den Ursprung. Hält Daten als Grid.\n  ACHTUNG: Nur bei reiner Translation (keine Rotation!) verwenden."
            }
        }, self)
        if dlg.exec(): 
            d = dlg.get_data()
            self.run_task(batch_transform_dataset_crs, src_crs=d['src'], tgt_crs=d['tgt'], grid_mode=d['mode'], debug=False)

    def prompt_merge_npz(self):
        if not self.check_ready(): return
        fname, _ = QFileDialog.getSaveFileName(self, "Dateiname", self.current_output_dir, "NPZ (*.npz)")
        if fname:
            self.console.appendPlainText("\n--- Merge NPZ ---\n")
            self.set_ui_running("Merge NPZ")
            self.thread = WorkerThread(merge_npz_datasets, self.current_input_dir, fname)
            self.thread.log_signal.connect(self.log_message)
            self.thread.finished_signal.connect(self.on_task_success)
            self.thread.start()

    def prompt_meshing(self):
        dlg = ParameterDialog("Meshing Konfiguration", {
            # --- MAIN SETTINGS ---
            'combine': {
                'type': bool, 'default': True, 'label': 'Kombinieren?',
                'help': "Fügt alle Meshes am Ende zu einer Datei zusammen (`combined_filename`)."
            },
            'filename': {
                'type': str, 'default': 'combined_model.ply', 'label': 'Dateiname',
                'help': "Dateiname für das zusammengefügte Mesh (nur relevant,\nwenn `combine=True`)."
            },
            'close artifacts': {
                'type': bool, 'default': False, 'label': 'Mögliche Artifakte reperieren?',
                'help': "Beim kombinieren von Kacheln kann es an den Ecken zu Fehlern in form von kleinen fehlenden Dreiecken kommen. Diese werden, wenn True, geschlossen."
            },
            'algo': {
                'type': 'combo', 'options': ['poisson', 'bpa'], 'label': 'Algo (Raw)',
                'help': "RAW: Der Meshing-Algorithmus\n- 'poisson': (Empfohlen) Screened Poisson Reconstruction. Erzeugt wasserdichte,\n  glatte Oberflächen. Gut für unvollständige Daten.\n- 'bpa': Ball-Pivoting Algorithm. Verbindet Punkte direkt. Erhält Details besser,\n  hinterlässt aber Löcher, wenn die Punktdichte zu gering ist."
            },
            'scan_type': {
                'type': 'combo', 'options': ['aerial', 'terrestrial', 'auto'], 'label': 'Typ',
                'help': "RAW: Art der Datenaufnahme zur Orientierung der Normalen.\n- 'aerial': Für Luftbilder/Drohnen - ALS (Normalen zeigen nach oben).\n- 'terrestrial': Für Boden-Scans - TLS.\n- 'auto': Versucht, anhand der Bounding-Box das Format zu erraten."
            },
            
            # --- POISSON SETTINGS ---
            'depth': {
                'type': int, 'default': 10, 'label': 'Tiefe (Depth)',
                'help': "RAW & poisson: Octree-Tiefe für den Poisson-Algorithmus. Steuert die Auflösung.\nRAM verbrauch ist exponential mit der depth.\n- 8-9: Grob, sehr schnell.\n- 10: Guter Standard für Gelände.\n- 11-12: Hohe Details, sehr hoher RAM-Verbrauch."
            },
            'weight': {
                'type': float, 'default': 6.0, 'label': 'Point Weight',
                'help': "RAW & poisson: Spezifiziert die Wichtigkeit der Messpunkte in der Reconstruction.\n- 0-4: Mehr glättung\n- 4-10: Algorithmus hält sich strenger an die Messpunkte; weniger smooth."
            },
            'samples': {
                'type': float, 'default': 1.0, 'label': 'Samples/Node',
                'help': "RAW & poisson: Spezifiziert die Mindestanzahl von Messpunkten, die innerhalb eines Octree-Knotens liegen müssen, damit er berücksichtigt wird.\n- < 1: Rauschfreie Messungen\n- 1-5: Für Messungen mit wenig Rauschen\n- 15-20: Für sehr rauschige Messungen"
            },

            # --- GRID / STITCHING ---
            'gap': {
                'type': float, 'default': 1.5, 'label': 'Gap Thresh [m]',
                'help': "GRID & Combine - Post Processing: Toleranzradius in Modelleinheiten für das Schließen von Lücken\n(Stitching) zwischen Kacheln.\nPunkte an den Rändern, die horizontal (X/Y) näher als dieser Wert beieinander liegen,\nwerden zusammengezogen (Vertex-Snapping), um ein wasserdichtes Modell zu garantieren."
            },

            # --- POST PROCESSING ---
            'smooth': {
                'type': bool, 'default': False, 'label': 'HC Smoothing',
                'help': "RAW & poisson - Post Processing: Aktiviert Laplace Filter. Relativ starker smoothing effekt."
            },
            'decimate': {
                'type': bool, 'default': False, 'label': 'Decimation',
                'help': "GRID or RAW - Post Processing: Aktiviert Quadric Edge Collapse Decimation. Verringert Anzahl der Faces durch Vereinachung des meshes."
            },
            'percent': {
                'type': float, 'default': 0.9, 'label': 'Target Ratio',
                'help': "GRID or RAW - Post Processing: Prozentuale anzahl der originalen Faces die nach der Vereinfachung noch vorhanden sein soll."
            },
            
            # --- SYSTEM ---
            'debug': {
                'type': bool, 'default': False, 'label': 'Debug (Single-Core)',
                'help': "Deaktiviert Multi-Processing (Parallelisierung). Führt alles sequenziell aus. Modus zum Debuging"
            }
        }, self)
        
        if dlg.exec():
            d = dlg.get_data()
            self.run_task(batch_process_npz_to_meshes,
                          combine=d['combine'], 
                          combined_filename=d['filename'], 
                          algorithm=d['algo'], 
                          scan_type=d['scan_type'], 
                          depth=d['depth'], 
                          pointweight=d['weight'], 
                          samplespernode=d['samples'], 
                          hc_laplacian_smoothing=d['smooth'], 
                          decimation=d['decimate'], 
                          percentile_of_faces=d['percent'], 
                          debug=d['debug'],
                          gap_threshold=d['gap'], 
                          close_artifacts=d['close artifacts'])

    def prompt_walls(self):
        if not self.current_input_dir: return
        f, _ = QFileDialog.getOpenFileName(self, "Mesh wählen", self.current_input_dir, "PLY (*.ply)")
        if f:
            dlg = ParameterDialog("Wände generieren", {
                'type': {
                    'type': 'combo', 'options': ['complex', 'rectangular'], 'label': 'Rand',
                    'help': "'rectangular': Erzwingt eine rechteckige Box (gut für Kacheln).\n'complex': Folgt dem exakten Rand des Meshes."
                }
            }, self)
            if dlg.exec():
                self.console.appendPlainText(f"\n--- Wände für: {os.path.basename(f)} ---\n")
                self.set_ui_running("Wände Gen.")
                self.thread = WorkerThread(batch_generate_mesh_enclosures, f, self.current_output_dir, boundary_type=dlg.get_data()['type'])
                self.thread.log_signal.connect(self.log_message)
                self.thread.finished_signal.connect(self.on_task_success)
                self.thread.start()

    def prompt_combine_ply(self):
        if not self.current_input_dir: return
        f1, _ = QFileDialog.getOpenFileName(self, "PLY 1", self.current_input_dir, "PLY (*.ply)")
        if not f1: return
        f2, _ = QFileDialog.getOpenFileName(self, "PLY 2", os.path.dirname(f1), "PLY (*.ply)")
        if not f2: return
        out, _ = QFileDialog.getSaveFileName(self, "Output", self.current_output_dir, "PLY (*.ply)")
        if out:
            self.console.appendPlainText("--- Combine PLY ---\n")
            self.set_ui_running("Combine PLY")
            self.thread = WorkerThread(concatenate_ply_files_memory_efficiently, f1, f2, out)
            self.thread.log_signal.connect(self.log_message)
            self.thread.finished_signal.connect(self.on_task_success)
            self.thread.start()

    def prompt_inspect_npz(self):
        f, _ = QFileDialog.getOpenFileName(self, "NPZ Datei", self.current_input_dir, "NPZ (*.npz)")
        if f:
            self.console.appendPlainText(f"\n--- Inspect: {os.path.basename(f)} ---\n")
            self.thread = WorkerThread(print_npz_metadata_structure, f)
            self.thread.log_signal.connect(self.log_message)
            self.thread.finished_signal.connect(self.on_task_success)
            self.thread.start()

    # --- VISUALISIERUNG (PROCESSED) ---

    def launch_scatter(self):
        if not self.check_ready(): return
        dlg = ParameterDialog("Scatter Viewer", {
            'res_x': {'type': int, 'default': 1000, 'label': 'Res X'},
            'res_y': {'type': int, 'default': 1000, 'label': 'Res Y'},
            'full': {'type': bool, 'default': False, 'label': 'Full Res (Slow)', 'help': 'Lädt alle Punkte (Vorsicht RAM).'},
            'sat': {'type': bool, 'default': False, 'label': 'RGB nutzen', 'help': 'Zeigt echte Farben falls vorhanden.'},
            'axes': {'type': bool, 'default': True, 'label': 'Achsen'}
        }, self)
        
        if dlg.exec():
            d = dlg.get_data()
            self.log_message(f"Starte Scatter (Process)...")
            self.set_ui_running("Scatter Viewer")
            self.current_process = multiprocessing.Process(
                target=render_interactive_3d_scatter_plot, 
                args=(self.current_input_dir, d['res_x'], d['res_y'], d['full'], d['sat'], d['axes'])
            )
            self.current_process.start()

    def launch_surface(self):
        if not self.check_ready(): return
        dlg = ParameterDialog("Surface Viewer", {
            'res_x': {'type': int, 'default': 1000, 'label': 'Res X'},
            'res_y': {'type': int, 'default': 1000, 'label': 'Res Y'},
            'native': {'type': bool, 'default': False, 'label': 'Native Res', 'help': 'Nutzt die Auflösung der Quelldaten.'},
            'axes': {'type': bool, 'default': True, 'label': 'Achsen'}
        }, self)
        
        if dlg.exec():
            d = dlg.get_data()
            self.log_message(f"Starte Surface (Process)...")
            self.set_ui_running("Surface Viewer")
            self.current_process = multiprocessing.Process(
                target=render_interactive_3d_surface_plot,
                args=(self.current_input_dir, d['res_x'], d['res_y'], d['native'], d['axes'])
            )
            self.current_process.start()

    def launch_plot_2d(self):
        if not self.check_ready(): return
        dlg = ParameterDialog("2D Plot", {
            'max': {
                'type': int, 'default': 8000, 'label': 'Max Pixel', 
                'help': 'Maximale Pixel/Punkte pro Achse für die Anzeige.\nVerhindert Memory-Overflows bei riesigen Daten.'
            }
        }, self)
        if dlg.exec():
            d = dlg.get_data()
            self.log_message("Starte 2D Plot...")
            self.set_ui_running("2D Plot")
            self.current_process = multiprocessing.Process(
                target=render_2d_elevation_plot, 
                args=(self.current_input_dir, d['max'])
            )
            self.current_process.start()
