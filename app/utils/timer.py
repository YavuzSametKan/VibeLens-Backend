import time
from colorama import Fore, Style, init

# Windows terminal renklerini düzeltir
init(autoreset=True)

class ExecutionTimer:
    """
    Kod bloklarının çalışma süresini ölçen Context Manager.
    """

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.time()
        print(f"{Fore.CYAN}⏳ [BAŞLIYOR] {self.step_name}...{Style.RESET_ALL}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time

        # Süreye göre renk değiştir (Yavaşsa Kırmızı, Hızlıysa Yeşil)
        color = Fore.GREEN
        if duration > 2.0: color = Fore.YELLOW
        if duration > 5.0: color = Fore.RED

        print(f"{color}✅ [BİTTİ]    {self.step_name} -> {duration:.4f} sn{Style.RESET_ALL}")