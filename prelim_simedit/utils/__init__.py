"""Utilities for the editing-cycle simulation.

Submodules are imported lazily by the stage runners to avoid pulling heavy
deps (torch / openai) unless that stage needs them.
"""
