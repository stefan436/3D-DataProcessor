# src/main.py

print("Starting Imports...", flush=True)

import os
import sys
import multiprocessing

import matplotlib
matplotlib.use('QtAgg') 
os.environ["VISPY_BACKEND"] = "pyqt6"
os.environ['QT_API'] = 'pyqt6'
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from config.settings import APP_STYLE
from gui.main_window import HeightProfileApp, apply_native_dark_palette

if __name__ == "__main__":
    print("Starting HeightProfilesApp...", flush=True)
    # Unter Linux ist 'spawn' viel sicherer für GUI-Apps als 'fork'
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    multiprocessing.freeze_support()

    # High DPI Settings
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)
    
    # 1. Native Palette setzen (ersetzt qdarktheme)
    apply_native_dark_palette(app)
    
    # 2. Deinen eigenen Style (APP_STYLE) darüberlegen
    app.setStyleSheet(APP_STYLE)
    
    try:
        window = HeightProfileApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"!!! CRASH BEI GUI START: {e}", flush=True)
        import traceback
        traceback.print_exc()
