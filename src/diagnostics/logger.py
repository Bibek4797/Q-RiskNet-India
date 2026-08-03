import time
import traceback
import streamlit as st

def log_event(level, message, elapsed_time=None):
    """
    Logs an event with a timestamp and details. Stores it in Streamlit session state if available.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = f" ({elapsed_time:.3f}s)" if elapsed_time is not None else ""
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": f"{message}{elapsed_str}",
        "traceback": None
    }
    
    print(f"[{level}] {timestamp} - {message}{elapsed_str}")
    try:
        if 'diagnostics_logs' not in st.session_state:
            st.session_state['diagnostics_logs'] = []
        st.session_state['diagnostics_logs'].append(log_entry)
    except Exception:
        pass

def log_info(message, elapsed_time=None):
    log_event("INFO", message, elapsed_time)

def log_warning(message):
    log_event("WARNING", message)

def log_error(message, error_exception=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    tb_str = "".join(traceback.format_exception(type(error_exception), error_exception, error_exception.__traceback__)) if error_exception else None
    
    log_entry = {
        "timestamp": timestamp,
        "level": "ERROR",
        "message": message,
        "traceback": tb_str
    }
    
    print(f"[ERROR] {timestamp} - {message}")
    if tb_str:
        print(tb_str)
        
    try:
        if 'diagnostics_logs' not in st.session_state:
            st.session_state['diagnostics_logs'] = []
        st.session_state['diagnostics_logs'].append(log_entry)
    except Exception:
        pass


class DiagnosticTimer:
    """
    A context manager to automatically time and log operations.
    """
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        log_info(f"Starting operation: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type:
            log_error(f"Failed operation: {self.operation_name} (Error: {exc_val})", exc_val)
        else:
            log_info(f"Completed operation: {self.operation_name}", elapsed)
        return False
