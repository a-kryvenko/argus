"""Compatibility command namespace; implementation is private."""
from importlib import import_module

__path__ = [import_module("notebooks.lib.development").command_path('revalidation')]
