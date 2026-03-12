# src/gui/dialogs.py

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
                             QDialogButtonBox, QScrollArea, QWidget, QFormLayout, QFrame, QApplication)

class ParameterDialog(QDialog):
    """
    Dynamischer Dialog zur Abfrage von Parametern.
    """
    def __init__(self, title, fields, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fields = fields
        self.inputs = {}
        
        self.setMinimumWidth(700)
        self.setStyleSheet("""
            QScrollArea { border: none; background-color: transparent; }
            QWidget#ContentWidget { background-color: transparent; }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { min-height: 25px; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        header_lbl = QLabel(title.upper())
        header_lbl.setStyleSheet("font-weight: bold; font-size: 12pt; color: #4a6572; margin-bottom: 5px;")
        main_layout.addWidget(header_lbl)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #444;")
        main_layout.addWidget(line)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True) 
        scroll.setFrameShape(QFrame.Shape.NoFrame) 
        
        content_widget = QWidget()
        content_widget.setObjectName("ContentWidget")
        form_layout = QFormLayout(content_widget)
        form_layout.setVerticalSpacing(20) 
        form_layout.setHorizontalSpacing(20)
        form_layout.setContentsMargins(5, 5, 15, 5)

        for key, config in fields.items():
            ftype = config.get('type', str)
            label_text = config.get('label', key)
            default = config.get('default', None)
            help_text = config.get('help', None)
            
            if ftype == 'combo':
                widget = QComboBox()
                widget.addItems(config.get('options', []))
                if default: widget.setCurrentText(str(default))
            elif ftype == float:
                widget = QLineEdit()
                if default is not None: widget.setText(str(default))
                widget.setPlaceholderText("0.0")
            elif ftype == int:
                widget = QSpinBox()
                widget.setRange(0, 2147483647)
                if default is not None: widget.setValue(int(default))
            elif ftype == bool:
                widget = QCheckBox()
                if default is not None: widget.setChecked(bool(default))
            else:
                widget = QLineEdit()
                if default is not None: widget.setText(str(default))
            
            if help_text:
                widget.setToolTip(help_text)

            self.inputs[key] = widget
            lbl_widget = QLabel(f"{label_text}:")
            lbl_widget.setStyleSheet("font-weight: bold; font-size: 10pt;")
            form_layout.addRow(lbl_widget, widget)
            
            if help_text:
                lbl_help = QLabel(help_text)
                lbl_help.setStyleSheet("color: #888; font-style: normal; font-size: 9pt; margin-bottom: 5px;")
                lbl_help.setWordWrap(True) 
                form_layout.addRow("", lbl_help)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setStyleSheet("color: #444;")
        main_layout.addWidget(line2)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_ok = buttons.button(QDialogButtonBox.StandardButton.Ok)
        btn_ok.setText("Anwenden")
        btn_ok.setStyleSheet("background-color: #4a6572; color: white; border: none; font-weight: bold; padding: 6px 15px;")
        
        btn_cancel = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        btn_cancel.setText("Abbrechen")

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        screen = QApplication.primaryScreen().availableGeometry()
        max_height = int(screen.height() * 0.85)
        self.resize(650, min(content_widget.sizeHint().height() + 180, max_height))

    def get_data(self):
        data = {}
        decimal_sep = "."
        if self.parent() and hasattr(self.parent(), 'combo_dec_sep'):
            decimal_sep = self.parent().combo_dec_sep.currentData()

        for key, widget in self.inputs.items():
            target_type = self.fields[key].get('type', str)

            if isinstance(widget, QComboBox):
                data[key] = widget.currentText()
            elif isinstance(widget, QCheckBox):
                data[key] = widget.isChecked()
            elif isinstance(widget, QSpinBox):
                data[key] = widget.value()
            else:
                text_val = widget.text().strip()
                if target_type == float:
                    try:
                        if not text_val: data[key] = 0.0
                        else:
                            if decimal_sep == ",":
                                clean_val = text_val.replace('.', '').replace(',', '.')
                            else:
                                clean_val = text_val.replace(',', '')
                            data[key] = float(clean_val)
                    except ValueError:
                        print(f"Warnung: Konnte '{text_val}' nicht in Zahl wandeln.")
                        data[key] = 0.0
                else:
                    data[key] = text_val
        return data
