# src/gui/workers.py

import sys
import io

from PyQt6.QtCore import QThread, pyqtSignal

class StreamRedirector(io.StringIO):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def write(self, text):
        if not text: return
        is_replace = '\r' in text
        clean_text = text.replace('\r', '').rstrip()
        if clean_text:
            self.signal.emit(clean_text, is_replace)

    def flush(self): pass

class WorkerThread(QThread):
    log_signal = pyqtSignal(str, bool) 
    finished_signal = pyqtSignal()
    
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = StreamRedirector(self.log_signal)
        sys.stderr = StreamRedirector(self.log_signal)
        try:
            import gc
            gc.collect()
            self.function(*self.args, **self.kwargs)
        except Exception as e:
            self.log_signal.emit(f"\n!!! ERROR: {str(e)}\n", False)
            import traceback
            self.log_signal.emit(traceback.format_exc(), False)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.finished_signal.emit()
