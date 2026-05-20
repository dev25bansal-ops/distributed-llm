"""Hardware detection across all architectures.

Probes available compute devices (CUDA, ROCm, MPS, XPU, CPU)
and returns normalized DeviceSpec objects.
"""

from __future__ import annotations

import platform

from loguru import logger

from distllm.core.hardware.device import DeviceSpec, DeviceType


class HardwareDetector:
    """Detects available hardware accelerators and CPU capabilities.

    Probes all device types and returns a list of DeviceSpec objects
    with normalized attributes across architectures.

    Usage:
        detector = HardwareDetector()
        all_devices = detector.detect_all()
        for dev in all_devices:
            print(dev.summary())
    """

    def detect_all(self) -> list[DeviceSpec]:
        """Detect all available compute devices across all architectures.

        Returns a list of DeviceSpec sorted by priority: accelerators
        first (CUDA, ROCm, MPS, XPU), then CPU.

        Returns:
            List of DeviceSpec for all detected devices.
        """
        devices: list[DeviceSpec] = []

        try:
            devices.extend(self.detect_cuda())
        except Exception as e:
            logger.debug(f"CUDA detection skipped: {e}")

        if not devices:
            try:
                devices.extend(self.detect_rocm())
            except Exception as e:
                logger.debug(f"ROCm detection skipped: {e}")

        if not devices:
            try:
                devices.extend(self.detect_mps())
            except Exception as e:
                logger.debug(f"MPS detection skipped: {e}")

        if not devices:
            try:
                devices.extend(self.detect_xpu())
            except Exception as e:
                logger.debug(f"XPU detection skipped: {e}")

        devices.extend(self.detect_cpu())

        logger.debug(f"Detected {len(devices)} device(s): {[d.summary() for d in devices]}")
        return devices

    def get_primary_device(self) -> DeviceSpec:
        """Get the primary (best available) compute device.

        Returns the first accelerator if available, or CPU.

        Returns:
            DeviceSpec for the primary device.
        """
        devices = self.detect_all()
        for dev in devices:
            if dev.is_accelerator:
                return dev
        return devices[0] if devices else DeviceSpec(
            device_type=DeviceType.CPU, device_id=0, name="unknown",
        )

    def detect_cuda(self) -> list[DeviceSpec]:
        """Detect NVIDIA CUDA-capable GPUs.

        Returns:
            List of DeviceSpec for CUDA devices (empty if not available).
        """
        try:
            import torch
        except ImportError:
            return []

        if not torch.cuda.is_available():
            return []

        devices: list[DeviceSpec] = []
        count = torch.cuda.device_count()

        for i in range(count):
            try:
                props = torch.cuda.get_device_properties(i)
                free_mem = props.total_memory - torch.cuda.memory_allocated(i)
                cap = (props.major, props.minor) if hasattr(props, 'major') else None
                devices.append(DeviceSpec(
                    device_type=DeviceType.CUDA,
                    device_id=i,
                    name=props.name,
                    total_memory_bytes=props.total_memory,
                    free_memory_bytes=free_mem,
                    compute_capability=cap,
                    backend="pytorch",
                    sm_count=getattr(props, 'multi_processor_count', 0),
                    max_threads_per_sm=getattr(props, 'max_threads_per_multi_processor', 0),
                    clock_rate_mhz=getattr(props, 'clock_rate', 0) // 1000,
                ))
            except Exception as e:
                logger.debug(f"Error probing CUDA device {i}: {e}")

        return devices

    def detect_rocm(self) -> list[DeviceSpec]:
        """Detect AMD ROCm-capable GPUs.

        Checks for AMD GPUs via torch.cuda when ROCm is the backend.

        Returns:
            List of DeviceSpec for ROCm devices (empty if not available).
        """
        try:
            import torch
        except ImportError:
            return []

        # ROCm uses torch.cuda with torch.version.hip set
        is_rocm = getattr(torch, 'version', None) and hasattr(torch.version, 'hip') and torch.version.hip is not None

        if not is_rocm or not torch.cuda.is_available():
            return []

        devices: list[DeviceSpec] = []
        count = torch.cuda.device_count()

        for i in range(count):
            try:
                props = torch.cuda.get_device_properties(i)
                free_mem = props.total_memory - torch.cuda.memory_allocated(i)
                devices.append(DeviceSpec(
                    device_type=DeviceType.ROCM,
                    device_id=i,
                    name=f"AMD {props.name}",
                    total_memory_bytes=props.total_memory,
                    free_memory_bytes=free_mem,
                    backend="pytorch",
                    sm_count=getattr(props, 'multi_processor_count', 0),
                    clock_rate_mhz=getattr(props, 'clock_rate', 0) // 1000,
                ))
            except Exception as e:
                logger.debug(f"Error probing ROCm device {i}: {e}")

        return devices

    def detect_mps(self) -> list[DeviceSpec]:
        """Detect Apple Silicon (MPS / Metal) devices.

        Returns:
            List of DeviceSpec for MPS devices (empty if not available).
        """
        try:
            import torch
        except ImportError:
            return []

        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            return []

        try:
            total_mem = 0
            free_mem = 0
            try:
                import psutil
                mem = psutil.virtual_memory()
                total_mem = mem.total
                free_mem = mem.available
            except ImportError:
                pass

            device_name = f"Apple {platform.machine()}"
            return [DeviceSpec(
                device_type=DeviceType.MPS,
                device_id=0,
                name=device_name,
                total_memory_bytes=total_mem,
                free_memory_bytes=free_mem,
                backend="pytorch",
            )]
        except Exception as e:
            logger.debug(f"MPS detection error: {e}")

        return []

    def detect_xpu(self) -> list[DeviceSpec]:
        """Detect Intel XPU/oneAPI devices.

        Checks for Intel GPUs via the XPU PyTorch backend.

        Returns:
            List of DeviceSpec for XPU devices (empty if not available).
        """
        try:
            import torch
        except ImportError:
            return []

        has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available() if hasattr(torch, 'xpu') else False

        if not has_xpu:
            return []

        devices: list[DeviceSpec] = []
        try:
            count = torch.xpu.device_count()
            for i in range(count):
                name = torch.xpu.get_device_name(i)
                total_mem = 0
                free_mem = 0
                try:
                    total_mem = torch.xpu.get_device_properties(i).total_memory
                except Exception:
                    pass
                devices.append(DeviceSpec(
                    device_type=DeviceType.XPU,
                    device_id=i,
                    name=name,
                    total_memory_bytes=total_mem,
                    free_memory_bytes=free_mem,
                    backend="pytorch",
                ))
        except Exception as e:
            logger.debug(f"XPU detection error: {e}")

        return devices

    def detect_cpu(self) -> list[DeviceSpec]:
        """Detect CPU capabilities.

        Creates a CPU DeviceSpec with available memory.

        Returns:
            List with a single DeviceSpec for the CPU.
        """
        total_mem = 0
        free_mem = 0
        cpu_count = 0
        cpu_name = ""

        try:
            import psutil
            mem = psutil.virtual_memory()
            total_mem = mem.total
            free_mem = mem.available
            cpu_count = psutil.cpu_count(logical=True) or 0
            cpu_name = platform.processor() or platform.machine()
        except ImportError:
            pass

        return [DeviceSpec(
            device_type=DeviceType.CPU,
            device_id=0,
            name=cpu_name or "CPU",
            total_memory_bytes=total_mem,
            free_memory_bytes=free_mem,
            backend="llamacpp",
            sm_count=cpu_count,
        )]
