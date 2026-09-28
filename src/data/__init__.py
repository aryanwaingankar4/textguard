"""Data pipeline (P1): turns raw Time-MMD into clean, chronological train/val/test splits.

Splits are always chronological, never random — random splits leak future information
into training and invalidate every downstream result.
"""
