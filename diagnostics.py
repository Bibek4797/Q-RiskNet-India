import time
import traceback
import streamlit as st

def log_event(level, message, elapsed_time=None):
    """
    Logs an event with a timestamp and details. Stores it in Streamlit session state.
    """
    if 'diagnostics_logs' not in st.session_state:
        st.session_state['diagnostics_logs'] = []
        
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    elapsed_str = f" ({elapsed_time:.3f}s)" if elapsed_time is not None else ""
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "message": f"{message}{elapsed_str}",
        "traceback": None
    }
    
    # Also print to standard output/console for backend visibility
    print(f"[{level}] {timestamp} - {message}{elapsed_str}")
    
    st.session_state['diagnostics_logs'].append(log_entry)

def log_info(message, elapsed_time=None):
    log_event("INFO", message, elapsed_time)

def log_warning(message):
    log_event("WARNING", message)

def log_error(message, error_exception=None):
    if 'diagnostics_logs' not in st.session_state:
        st.session_state['diagnostics_logs'] = []
        
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
        
    st.session_state['diagnostics_logs'].append(log_entry)

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
        # Return False to let any exceptions propagate, or we handle them in code
        return False
