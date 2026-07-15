import psutil
from .logging import log

def log_system_metrics(stage: str = "System Metrics"):
    """Log current RAM and VRAM usage."""
    try:
        # RAM usage
        process = psutil.Process()
        ram_info = process.memory_info()
        ram_usage_mb = ram_info.rss / (1024 * 1024)
        total_ram = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        
        metrics_msg = f"RAM Usage: {ram_usage_mb:.2f} MB / {total_ram:.2f} GB"
        
        # Try to get VRAM usage if torch is available
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                metrics_msg += f" | VRAM Allocated: {allocated:.2f} MB | VRAM Reserved: {reserved:.2f} MB"
        except ImportError:
            pass
            
        log(metrics_msg, title=stage, style="bold magenta")
    except Exception as e:
        log(f"Failed to log metrics: {e}", title="Metrics Error", style="bold red")
