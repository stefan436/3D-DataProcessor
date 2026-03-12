# src/main.py

import os
import sys
import multiprocessing

import matplotlib
matplotlib.use('QtAgg') 
os.environ["VISPY_BACKEND"] = "pyqt6"
os.environ['QT_API'] = 'pyqt6'
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import qdarktheme

from config.settings import APP_STYLE
from gui.main_window import HeightProfileApp

if __name__ == "__main__":
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
    
    qdarktheme.setup_theme(theme="dark", corner_shape="sharp")
    
    app.setStyleSheet(app.styleSheet() + APP_STYLE)
    
    try:
        window = HeightProfileApp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"!!! CRASH BEI GUI START: {e}", flush=True)
        import traceback
        traceback.print_exc()
