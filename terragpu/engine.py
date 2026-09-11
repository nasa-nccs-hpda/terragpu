"""Explicit backend selection and optional distributed execution."""
import os
from importlib import import_module
from types import ModuleType


def array_module(xp=None):
    """Select numpy/cupy; auto falls back to CPU, explicit cupy fails clearly."""
    if isinstance(xp, ModuleType):
        return xp
    name = xp or os.environ.get("ARRAY_MODULE", "auto")
    if name == "numpy":
        return import_module("numpy")
    if name not in {"auto", "cupy"}:
        raise ValueError(f"ARRAY_MODULE={name} not known")
    try:
        cp = import_module("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA devices found")
        return cp
    except (ImportError, OSError, RuntimeError) as exc:
        if name == "cupy":
            raise RuntimeError("CuPy requires a compatible installation and working CUDA device") from exc
        return import_module("numpy")


def configure_dask(local_directory=None, n_workers=None, device="gpu", **kwargs):
    """Return a Client owning a local cluster. Caller must close client and cluster.

    Extra keyword arguments go to LocalCluster or LocalCUDACluster (e.g.
    threads_per_worker, device_memory_limit, rmm_pool_size).
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")
    try:
        from distributed import Client, LocalCluster
    except ImportError as exc:
        raise ImportError("Install terragpu[parallel] for Dask clusters") from exc
    cluster_type = LocalCluster
    if device == "gpu":
        array_module("cupy")
        try:
            from dask_cuda import LocalCUDACluster
        except ImportError as exc:
            raise RuntimeError("GPU clusters require dask-cuda; see requirements/README.md") from exc
        cluster_type = LocalCUDACluster
        kwargs.setdefault("device_memory_limit", 0.8)
    cluster = cluster_type(local_directory=local_directory, n_workers=n_workers, **kwargs)
    try:
        return Client(cluster)
    except Exception:
        cluster.close()
        raise
