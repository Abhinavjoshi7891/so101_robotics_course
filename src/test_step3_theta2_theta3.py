#!/usr/bin/env python3
"""
ECE4560 – SO-101 Assignment
===========================

TEST SCRIPT FOR θ₂ AND θ₃ (PLANAR GEOMETRIC IK)

This script tests θ₂ and θ₃ using the SAME simplified planar model
assumed in the assignment handout.

IMPORTANT:
- This script does NOT use MuJoCo FK.
- This script validates IK against a MATCHING planar FK.
- This is exactly how the assignment geometry is meant to be tested.

Author: (you)
"""

import numpy as np
from math import sin, cos, atan2, sqrt, acos

# ==============================================================
# Robot parameters (instructional model)
# ==============================================================

# Upper arm length (shoulder → elbow)
L2 = 0.11257 + 0.028     # meters

# Forearm length (elbow → wrist)
L3 = 0.1349              # meters

# ==============================================================
# Test target (planar wrist position)
# ==============================================================

# These are PLANAR coordinates after θ₁ reduction
# (x forward, z vertical)
x_target = 0.12      # meters
z_target = -0.05     # meters

print("="*70)
print("SO-101 ASSIGNMENT TEST — θ₂, θ₃ (PLANAR MODEL)")
print("="*70)

print(f"\nTarget wrist position (planar):")
print(f"  x = {x_target:.4f} m")
print(f"  z = {z_target:.4f} m")

# ==============================================================
# Step 1: Distance to wrist
# ==============================================================

r = sqrt(x_target**2 + z_target**2)

print(f"\nDistance to wrist:")
print(f"  r = {r:.4f} m")

# ==============================================================
# Step 2: Solve θ₃ (elbow flex) — law of cosines
# ==============================================================

cos_theta3 = (r**2 - L2**2 - L3**2) / (2 * L2 * L3)
cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)

theta3 = acos(cos_theta3)   # elbow-down solution
theta3_deg = np.degrees(theta3)

print(f"\nθ₃ (elbow flex):")
print(f"  θ₃ = {theta3_deg:.2f}°")

# ==============================================================
# Step 3: Solve θ₂ (shoulder lift)
# ==============================================================

phi = atan2(z_target, x_target)
psi = atan2(
    L3 * sin(theta3),
    L2 + L3 * cos(theta3)
)

theta2 = phi - psi
theta2_deg = np.degrees(theta2)

print(f"\nθ₂ (shoulder lift):")
print(f"  θ₂ = {theta2_deg:.2f}°")

# ==============================================================
# Step 4: PLANAR Forward Kinematics (verification)
# ==============================================================

x_fk = L2 * cos(theta2) + L3 * cos(theta2 + theta3)
z_fk = L2 * sin(theta2) + L3 * sin(theta2 + theta3)

error = sqrt((x_fk - x_target)**2 + (z_fk - z_target)**2)

print("\nPLANAR FK VERIFICATION:")
print(f"  FK x = {x_fk:.4f} m")
print(f"  FK z = {z_fk:.4f} m")
print(f"  Error = {error:.6f} m")

# ==============================================================
# Verdict
# ==============================================================

print("\n" + "="*70)
if error < 1e-6:
    print("✅ SUCCESS: θ₂ and θ₃ are CORRECT for the assignment model")
else:
    print("❌ ERROR: IK does not match planar FK")
print("="*70)
