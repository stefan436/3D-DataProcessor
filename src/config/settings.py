# src/config/settings.py

import sys

# --- KONFIGURATION ---
NODATA_VALUE = -9999.0

# =================================================================
# FIX FÜR GUI START (Verknüpfung): Fehlende Konsole abfangen
# =================================================================
if sys.stderr is None or sys.stdout is None:
    class NullWriter:
        def write(self, text): pass
        def flush(self): pass
    
    # Wenn kein Output-Kanal da ist (None), ersetzen wir ihn durch den Dummy
    if sys.stderr is None: sys.stderr = NullWriter()
    if sys.stdout is None: sys.stdout = NullWriter()
    
# =================================================================
# STYLING CONSTANTS & QSS
# =================================================================
APP_STYLE = """
/* GLOBAL APP STYLING - Modern Classic Dark (Subtle Edition) */
QMainWindow {
    background-color: #1e1e1e;
}

/* HEADER AREA */
QGroupBox#HeaderBox {
    background-color: #252526;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    margin-top: 5px;
    font-weight: bold;
    color: #cccccc;
}

QLineEdit {
    background-color: #333337;
    border: 1px solid #454545;
    border-radius: 2px;
    color: #f1f1f1;
    padding: 4px;
    selection-background-color: #4a6572;
}
QLineEdit:read-only {
    background-color: #2d2d30;
    color: #a0a0a0;
    font-style: italic;
}

/* BUTTONS */
QPushButton {
    background-color: #3e3e42;
    border: 1px solid #555555;
    color: #f1f1f1;
    padding: 5px 12px;
    border-radius: 3px;
    font-weight: normal;
}
QPushButton:hover {
    background-color: #505055;
    border-color: #4a6572;
}
QPushButton:pressed {
    background-color: #4a6572;
    color: white;
}

/* RIBBON TAB WIDGET */
QTabWidget::pane {
    border: 1px solid #3e3e42;
    background-color: #252526;
    top: -1px; 
}
QTabWidget::tab-bar {
    left: 5px; 
}
QTabBar::tab {
    background: #1e1e1e;
    color: #999999;
    padding: 8px 20px;
    border: 1px solid #3e3e42;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    min-width: 80px;
}
QTabBar::tab:selected {
    background: #252526;
    color: #ffffff;
    border-bottom: 1px solid #252526; 
    border-top: 2px solid #4a6572;
    font-weight: bold;
}
QTabBar::tab:hover {
    background: #2d2d30;
    color: #eeeeee;
}

/* RIBBON TOOL BUTTONS */
QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
    color: #dddddd;
    font-size: 11px;
}
QToolButton:hover {
    background-color: #3e3e42;
    border: 1px solid #555555;
}
QToolButton:pressed {
    background-color: #4a6572;
    color: white;
}

/* CONSOLE */
QPlainTextEdit {
    background-color: #121212;
    border: 1px solid #3e3e42;
    color: #00ff00;
    selection-background-color: #4a6572;
    selection-color: #ffffff;
}
QScrollBar:vertical {
    background: #1e1e1e;
    width: 14px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #424242;
    min-height: 20px;
    border-radius: 7px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: #686868;
}

/* STATUS BAR */
QStatusBar {
    background-color: #252526;
    color: #cccccc;
    border-top: 1px solid #3e3e42;
}
QStatusBar QLabel {
    color: #cccccc;
    padding: 0 5px;
}

/* DIALOGS */
QDialog {
    background-color: #252526;
}
QLabel {
    color: #e1e1e1;
}
"""