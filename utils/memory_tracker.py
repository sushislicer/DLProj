"""
Memory tracking utility for monitoring GPU and system memory usage.
"""

import os
import time
import psutil
import torch
import logging
from typing import Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager


class MemoryTracker:
    """Track GPU and system memory usage during training/inference."""
    
    def __init__(self, log_dir: str = "logs", log_file: str = "memory_usage.log"):
        """
        Initialize memory tracker.
        
        Args:
            log_dir: Directory to save memory logs
            log_file: Name of the memory log file
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger.
        # Use a per-logfile logger name to prevent handler duplication when
        # multiple MemoryTracker instances are created in the same process.
        logger_name = f"MemoryTracker[{os.path.abspath(self.log_file)}]"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # Add handlers only once per logger.
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler(self.log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # Memory history
        self.history: List[Dict] = []
        self.start_time = None
        self.peak_gpu_memory = 0
        self.peak_system_memory = 0
    
    def get_gpu_memory(self) -> Dict[str, float]:
        """
        Get GPU memory usage.
        
        Returns:
            Dictionary with GPU memory statistics
        """
        if not torch.cuda.is_available():
            return {
                'allocated_mb': 0.0,
                'reserved_mb': 0.0,
                'free_mb': 0.0,
                'total_mb': 0.0,
                'percent_used': 0.0
            }
        
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        
        # Get total and free memory
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**2  # MB
        free = total - reserved
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'free_mb': free,
            'total_mb': total,
            'percent_used': (reserved / total) * 100
        }
    
    def get_system_memory(self) -> Dict[str, float]:
        """
        Get system memory usage.
        
        Returns:
            Dictionary with system memory statistics
        """
        mem = psutil.virtual_memory()
        
        return {
            'total_mb': mem.total / 1024**2,
            'available_mb': mem.available / 1024**2,
            'used_mb': mem.used / 1024**2,
            'percent_used': mem.percent,
            'free_mb': mem.free / 1024**2
        }
    
    def get_memory_stats(self) -> Dict:
        """
        Get comprehensive memory statistics.
        
        Returns:
            Dictionary with all memory statistics
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'gpu': self.get_gpu_memory(),
            'system': self.get_system_memory()
        }
        
        # Update peak values
        if stats['gpu']['allocated_mb'] > self.peak_gpu_memory:
            self.peak_gpu_memory = stats['gpu']['allocated_mb']
        
        if stats['system']['used_mb'] > self.peak_system_memory:
            self.peak_system_memory = stats['system']['used_mb']
        
        return stats
    
    def log_memory(self, stage: str = "", message: str = ""):
        """
        Log current memory usage.
        
        Args:
            stage: Current pipeline stage
            message: Additional message
        """
        stats = self.get_memory_stats()
        
        # Format log message
        log_msg = f"[{stage}] " if stage else ""
        log_msg += f"GPU: {stats['gpu']['allocated_mb']:.2f}MB / {stats['gpu']['total_mb']:.2f}MB "
        log_msg += f"({stats['gpu']['percent_used']:.1f}%) | "
        log_msg += f"System: {stats['system']['used_mb']:.2f}MB / {stats['system']['total_mb']:.2f}MB "
        log_msg += f"({stats['system']['percent_used']:.1f}%)"
        
        if message:
            log_msg += f" | {message}"
        
        self.logger.info(log_msg)
        
        # Add to history
        stats['stage'] = stage
        stats['message'] = message
        self.history.append(stats)
    
    def start_tracking(self):
        """Start memory tracking session."""
        self.start_time = time.time()
        self.peak_gpu_memory = 0
        self.peak_system_memory = 0
        self.history = []
        self.logger.info("=" * 80)
        self.logger.info("Memory Tracking Started")
        self.logger.info("=" * 80)
        self.log_memory("INIT", "Initial memory state")
    
    def stop_tracking(self):
        """Stop memory tracking and print summary."""
        if self.start_time is None:
            self.logger.warning("Memory tracking was not started")
            return
        
        elapsed_time = time.time() - self.start_time
        
        self.logger.info("=" * 80)
        self.logger.info("Memory Tracking Summary")
        self.logger.info("=" * 80)
        self.logger.info(f"Total tracking time: {elapsed_time:.2f}s")
        self.logger.info(f"Peak GPU memory: {self.peak_gpu_memory:.2f}MB")
        self.logger.info(f"Peak system memory: {self.peak_system_memory:.2f}MB")
        self.logger.info(f"Total memory snapshots: {len(self.history)}")
        self.logger.info("=" * 80)
        
        # Save history to file
        self.save_history()
    
    def save_history(self):
        """Save memory history to JSON file."""
        import json
        
        history_file = os.path.join(self.log_dir, "memory_history.json")
        with open(history_file, 'w') as f:
            json.dump({
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'peak_gpu_memory_mb': self.peak_gpu_memory,
                'peak_system_memory_mb': self.peak_system_memory,
                'history': self.history
            }, f, indent=2)
        
        self.logger.info(f"Memory history saved to {history_file}")
    
    def print_summary(self):
        """Print a formatted summary of memory usage."""
        if not self.history:
            print("No memory data available")
            return
        
        print("\n" + "=" * 80)
        print("MEMORY USAGE SUMMARY")
        print("=" * 80)
        
        # GPU summary
        gpu_stats = [h['gpu'] for h in self.history]
        avg_gpu_allocated = sum(s['allocated_mb'] for s in gpu_stats) / len(gpu_stats)
        avg_gpu_percent = sum(s['percent_used'] for s in gpu_stats) / len(gpu_stats)
        
        print(f"\nGPU Memory:")
        print(f"  Peak:     {self.peak_gpu_memory:.2f} MB")
        print(f"  Average:  {avg_gpu_allocated:.2f} MB")
        print(f"  Average %: {avg_gpu_percent:.1f}%")
        
        # System summary
        sys_stats = [h['system'] for h in self.history]
        avg_sys_used = sum(s['used_mb'] for s in sys_stats) / len(sys_stats)
        avg_sys_percent = sum(s['percent_used'] for s in sys_stats) / len(sys_stats)
        
        print(f"\nSystem Memory:")
        print(f"  Peak:     {self.peak_system_memory:.2f} MB")
        print(f"  Average:  {avg_sys_used:.2f} MB")
        print(f"  Average %: {avg_sys_percent:.1f}%")
        
        print("=" * 80 + "\n")
    
    @contextmanager
    def track_stage(self, stage_name: str):
        """
        Context manager for tracking a specific stage.
        
        Args:
            stage_name: Name of the stage being tracked
        
        Usage:
            with tracker.track_stage("PiSSA"):
                # Run PiSSA code
                pass
        """
        self.log_memory(stage_name, "Stage started")
        start_time = time.time()
        
        try:
            yield
            elapsed = time.time() - start_time
            self.log_memory(stage_name, f"Stage completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            self.log_memory(stage_name, f"Stage failed after {elapsed:.2f}s: {str(e)}")
            raise
    
    def get_memory_report(self) -> str:
        """
        Generate a detailed memory report.
        
        Returns:
            Formatted memory report string
        """
        if not self.history:
            return "No memory data available"
        
        report = []
        report.append("=" * 80)
        report.append("DETAILED MEMORY REPORT")
        report.append("=" * 80)
        
        # Overall statistics
        gpu_stats = [h['gpu'] for h in self.history]
        sys_stats = [h['system'] for h in self.history]
        
        report.append(f"\nTotal snapshots: {len(self.history)}")
        report.append(f"Peak GPU memory: {self.peak_gpu_memory:.2f} MB")
        report.append(f"Peak system memory: {self.peak_system_memory:.2f} MB")
        
        # Per-stage statistics
        stages = {}
        for h in self.history:
            stage = h.get('stage', 'UNKNOWN')
            if stage not in stages:
                stages[stage] = []
            stages[stage].append(h)
        
        report.append("\n\nPer-Stage Statistics:")
        report.append("-" * 80)
        
        for stage, snapshots in sorted(stages.items()):
            if stage == 'UNKNOWN':
                continue
            
            stage_gpu = [s['gpu'] for s in snapshots]
            stage_sys = [s['system'] for s in snapshots]
            
            avg_gpu = sum(s['allocated_mb'] for s in stage_gpu) / len(stage_gpu)
            max_gpu = max(s['allocated_mb'] for s in stage_gpu)
            avg_sys = sum(s['used_mb'] for s in stage_sys) / len(stage_sys)
            max_sys = max(s['used_mb'] for s in stage_sys)
            
            report.append(f"\n{stage}:")
            report.append(f"  Snapshots: {len(snapshots)}")
            report.append(f"  GPU - Avg: {avg_gpu:.2f} MB, Max: {max_gpu:.2f} MB")
            report.append(f"  System - Avg: {avg_sys:.2f} MB, Max: {max_sys:.2f} MB")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def print_memory_info():
    """Print current memory information."""
    tracker = MemoryTracker()
    
    print("\n" + "=" * 80)
    print("CURRENT MEMORY STATUS")
    print("=" * 80)
    
    # GPU memory
    gpu = tracker.get_gpu_memory()
    print(f"\nGPU Memory:")
    print(f"  Allocated: {gpu['allocated_mb']:.2f} MB")
    print(f"  Reserved:  {gpu['reserved_mb']:.2f} MB")
    print(f"  Free:      {gpu['free_mb']:.2f} MB")
    print(f"  Total:     {gpu['total_mb']:.2f} MB")
    print(f"  Used:      {gpu['percent_used']:.1f}%")
    
    # System memory
    sys = tracker.get_system_memory()
    print(f"\nSystem Memory:")
    print(f"  Total:     {sys['total_mb']:.2f} MB")
    print(f"  Available: {sys['available_mb']:.2f} MB")
    print(f"  Used:      {sys['used_mb']:.2f} MB")
    print(f"  Free:      {sys['free_mb']:.2f} MB")
    print(f"  Used:      {sys['percent_used']:.1f}%")
    
    print("=" * 80 + "\n")


if __name__ == '__main__':
    # Test the memory tracker
    print("Testing Memory Tracker...")
    print_memory_info()
    
    # Create a tracker and run a simple test
    tracker = MemoryTracker()
    tracker.start_tracking()
    
    # Simulate some work
    for i in range(5):
        time.sleep(0.5)
        tracker.log_memory("TEST", f"Iteration {i+1}")
    
    tracker.stop_tracking()
    tracker.print_summary()
    print(tracker.get_memory_report())
