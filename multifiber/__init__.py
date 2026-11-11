"""multifiber: extraction and preprocessing for the multi-site stim-record rig."""

__version__ = "0.1.0.dev0"

from . import (  # noqa: F401
    calibration, discovery, events, extract, io, matio, metadata, pipeline,
    preprocess, progress, variants,
)
from .pipeline import PipelineConfig, process_directory, process_session  # noqa: F401

__all__ = [
    "calibration", "discovery", "events", "extract", "io", "matio",
    "metadata", "pipeline", "preprocess", "progress", "variants",
    "PipelineConfig", "process_session", "process_directory", "__version__",
]
