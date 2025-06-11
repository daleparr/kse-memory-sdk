"""
KSE Memory SDK - Quickstart Module

Provides zero-configuration demo and benchmarking capabilities
for instant developer onboarding and "wow" moments.
"""

from .demo import QuickstartDemo
from .benchmark import BenchmarkRunner
from .datasets import SampleDatasets

__all__ = [
    "QuickstartDemo",
    "BenchmarkRunner", 
    "SampleDatasets",
]