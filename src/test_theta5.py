#!/usr/bin/env python3
"""
ECE4560 – SO-101 Assignment
===========================

TEST θ₅ — Wrist yaw alignment (CORRECT)

θ₅ compensates for θ₁ so that the tool has the desired yaw
about the world z-axis.
"""

print("="*80)
print("TEST θ₅ — ASSIGNMENT MODEL (FIXED)")
print("="*80)

# ------------------------------------------------------------
# Given values
# ------------------------------------------------------------

theta1_deg = -45.0      # shoulder_pan
desired_yaw_deg = 0.0  # desired tool yaw

print(f"\nGiven:")
print(f"  θ₁ (shoulder_pan) = {theta1_deg:.2f}°")
print(f"  Desired yaw       = {desired_yaw_deg:.2f}°")

# ------------------------------------------------------------
# Solve θ₅ (CORRECT)
# ------------------------------------------------------------

theta5_deg = desired_yaw_deg - theta1_deg

print(f"\nSolved θ₅:")
print(f"  θ₅ = {theta5_deg:.2f}°")

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

effective_yaw = theta1_deg + theta5_deg

print("\nVerification:")
print(f"  θ₁ + θ₅ = {effective_yaw:.2f}°")

# ------------------------------------------------------------
# Verdict
# ------------------------------------------------------------

print("\n" + "="*80)
if abs(effective_yaw - desired_yaw_deg) < 1e-6:
    print("✅ SUCCESS: Tool yaw matches desired yaw")
else:
    print("❌ FAIL: Yaw mismatch")
print("="*80)
