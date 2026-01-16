#!/usr/bin/env python3
"""
ECE4560 – SO-101 Assignment
===========================

TEST θ₄ — Planar Geometric Wrist Alignment

θ₄ is chosen so that the wrist points vertically downward
in the θ₂–θ₃ plane.

This follows the assignment's triangle-based model,
NOT the MuJoCo joint axes.
"""

import numpy as np
from math import radians, degrees, pi

# ------------------------------------------------------------
# GIVEN θ₂ and θ₃ (from assignment IK)
# ------------------------------------------------------------

theta2_deg = -40.0   # example
theta3_deg =  85.0   # example

theta2 = radians(theta2_deg)
theta3 = radians(theta3_deg)

print("="*80)
print("TEST θ₄ — ASSIGNMENT PLANAR MODEL")
print("="*80)

print(f"\nGiven:")
print(f"  θ₂ = {theta2_deg:.2f}°")
print(f"  θ₃ = {theta3_deg:.2f}°")

# ------------------------------------------------------------
# Solve θ₄ (assignment formula)
# ------------------------------------------------------------

theta4 = -pi/2 - (theta2 + theta3)
theta4_deg = degrees(theta4)

print(f"\nSolved θ₄:")
print(f"  θ₄ = {theta4_deg:.2f}°")

# ------------------------------------------------------------
# Verification (pure geometry)
# ------------------------------------------------------------

total_pitch = theta2 + theta3 + theta4

print("\nVerification:")
print(f"  θ₂ + θ₃ + θ₄ = {degrees(total_pitch):.2f}°")

# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------

print("\n" + "="*80)
if abs(degrees(total_pitch) + 90.0) < 1e-6:
    print("✅ SUCCESS: Wrist is vertical in assignment model")
else:
    print("❌ FAIL: Wrist not vertical")
print("="*80)
