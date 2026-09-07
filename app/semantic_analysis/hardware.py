from __future__ import annotations

import importlib.util
import os

import psutil

from app.semantic_analysis.models import HardwareCapability


def inspect_hardware() -> HardwareCapability:
    vm = psutil.virtual_memory()
    result = HardwareCapability(
        torch_available=importlib.util.find_spec("torch") is not None,
        system_ram_total_mb=int(vm.total / (1024 * 1024)),
        system_ram_available_mb=int(vm.available / (1024 * 1024)),
        cpu_count=os.cpu_count(),
    )
    if not result.torch_available:
        return result
    try:
        import torch

        result.cuda_available = bool(torch.cuda.is_available())
        result.cuda_device_count = int(torch.cuda.device_count()) if result.cuda_available else 0
        if result.cuda_available and result.cuda_device_count:
            props = torch.cuda.get_device_properties(0)
            result.gpu_name = str(props.name)
            result.gpu_total_memory_mb = int(props.total_memory / (1024 * 1024))
            try:
                free, _ = torch.cuda.mem_get_info(0)
                result.gpu_free_memory_mb = int(free / (1024 * 1024))
            except Exception:
                result.gpu_free_memory_mb = None
    except Exception:
        # Hardware introspection is best effort. Actual runtime loading returns typed errors.
        pass
    return result
