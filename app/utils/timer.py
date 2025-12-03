import time
from colorama import Fore, Style, init

# Initializes colorama for Windows terminal compatibility
init(autoreset=True)

class ExecutionTimer:
    """
    Context Manager to measure the execution time of code blocks.
    """

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        print(f"{Fore.CYAN}⏳ [STARTING] {self.step_name}...{Style.RESET_ALL}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time

        # Color based on duration (Green for fast, Red for slow)
        color = Fore.GREEN
        if duration > 2.0: color = Fore.YELLOW
        if duration > 5.0: color = Fore.RED

        print(f"{color}✅ [FINISHED] {self.step_name} -> {duration:.4f} sec{Style.RESET_ALL}")