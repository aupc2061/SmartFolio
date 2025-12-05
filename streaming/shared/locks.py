"""
Thread synchronization primitives for the streaming pipeline.
"""

import threading
from typing import Dict
from streaming.config import NUM_RISK_GROUPS


class StreamingLocks:
    """
    Centralized lock management for the streaming pipeline.
    
    Lock acquisition order to prevent deadlocks:
    1. stock_buffer_lock
    2. csv_write_lock
    3. finetune_counter_locks[i] (by index order)
    4. model_locks[i] (by index order)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure one set of locks across the application."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize_locks()
        return cls._instance
    
    def _initialize_locks(self):
        """Initialize all locks."""
        # Lock for accessing the in-memory stock data buffer
        self.stock_buffer_lock = threading.Lock()
        
        # Lock for writing monthly CSV files
        self.csv_write_lock = threading.Lock()
        self.csv_lock = self.csv_write_lock  # Alias for backward compatibility
        
        # Lock for accessing user database
        self.user_db_lock = threading.Lock()
        
        # Lock for finetune operations (global)
        self.finetune_lock = threading.Lock()
        
        # Locks for finetune counters (one per risk group)
        self.finetune_counter_locks: Dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(NUM_RISK_GROUPS)
        }
        
        # Locks for model access (one per risk group)
        self.model_locks: Dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(NUM_RISK_GROUPS)
        }
        
        # Condition variable for signaling new data available
        self.data_available = threading.Condition()
        
        # Event for graceful shutdown
        self.shutdown_event = threading.Event()
    
    def acquire_for_month_flush(self):
        """
        Acquire all locks needed for flushing monthly data.
        Should be used with care to prevent deadlocks.
        """
        self.stock_buffer_lock.acquire()
        self.csv_write_lock.acquire()
    
    def release_after_month_flush(self):
        """Release locks after flushing monthly data."""
        self.csv_write_lock.release()
        self.stock_buffer_lock.release()
    
    def increment_all_finetune_counters(self, counters: Dict[int, int]):
        """
        Safely increment all finetune counters.
        Must be called with stock_buffer_lock already held.
        """
        for i in range(NUM_RISK_GROUPS):
            with self.finetune_counter_locks[i]:
                counters[i] = counters.get(i, 0) + 1
    
    def signal_shutdown(self):
        """Signal all threads to shut down gracefully."""
        self.shutdown_event.set()
        with self.data_available:
            self.data_available.notify_all()
    
    def should_shutdown(self) -> bool:
        """Check if shutdown has been signaled."""
        return self.shutdown_event.is_set()


# Global singleton instance
locks = StreamingLocks()
