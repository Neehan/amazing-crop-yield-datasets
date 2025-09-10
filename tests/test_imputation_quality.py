#!/usr/bin/env python3
"""
Test script for analyzing crop calendar imputation quality.
Compares imputed vs non-imputed planted and harvested area patterns for wheat per admin 1.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def calculate_spread_stats(monthly_data, name):
    """Calculate spread statistics for monthly data"""
    # Find months with >1% activity
    active_months = monthly_data[monthly_data > 0.01]
    if len(active_months) == 0:
        return {
            "spread_months": 0,
            "peak_month": 0,
            "concentration": 0,
            "entropy": 0,
            "active_months": [],
        }

    # Number of months with significant activity
    spread_months = len(active_months)

    # Peak month (extract number from column name)
    peak_month = int(monthly_data.idxmax().split("_")[-1])

    # Concentration (how much is in the peak month)
    concentration = monthly_data.max()

    # Calculate entropy (measure of spread)
    # Remove zeros and normalize
    non_zero = monthly_data[monthly_data > 0]
    if len(non_zero) > 0:
        normalized = non_zero / non_zero.sum()
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
    else:
        entropy = 0

    # Get active month numbers
    active_month_nums = [int(col.split("_")[-1]) for col in active_months.index]

    return {
        "spread_months": spread_months,
        "peak_month": peak_month,
        "concentration": concentration,
        "entropy": entropy,
        "active_months": active_month_nums,
    }


def analyze_imputation_quality(crop_calendar_file, yield_file, setting_name="Current"):
    """Analyze imputation quality for wheat crop calendar data"""

    print(f"=== SPREAD/SEASONALITY ANALYSIS - {setting_name.upper()} ===")

    # Load data
    crop_calendar = pd.read_csv(crop_calendar_file)
    yield_data = pd.read_csv(yield_file)

    # Create uppercase column for matching
    crop_calendar["admin_level_1_upper"] = crop_calendar[
        "admin_level_1_name"
    ].str.upper()

    # Check all major wheat provinces
    major_wheat_provinces = [
        "Buenos Aires",
        "Córdoba",
        "Santa Fe",
        "La Pampa",
        "Entre Ríos",
    ]
    planted_cols = [f"planted_month_{i}" for i in range(1, 13)]
    harvested_cols = [f"harvested_month_{i}" for i in range(1, 13)]

    # Store results for comparison
    results = {}

    for province in major_wheat_provinces:
        print(f"\n--- {province.upper()} ---")

        # Get crop calendar data
        cc_data = crop_calendar[crop_calendar["admin_level_1_name"] == province].copy()

        if len(cc_data) > 0:
            # Check imputation status
            zero_area_mask = cc_data["total_area"] == 0.0
            monthly_cols = [col for col in cc_data.columns if "month_" in col]
            has_monthly_data = (cc_data[monthly_cols] > 0).any(axis=1)
            imputed_mask = zero_area_mask & has_monthly_data
            non_imputed_mask = ~imputed_mask

            print(
                f"Imputed records: {imputed_mask.sum()}, Non-imputed records: {non_imputed_mask.sum()}"
            )

            # Analyze patterns if we have both types
            if imputed_mask.sum() > 0 and non_imputed_mask.sum() > 0:
                imputed_data = cc_data[imputed_mask]
                non_imputed_data = cc_data[non_imputed_mask]

                # Calculate average monthly patterns
                imputed_planted_avg = imputed_data[planted_cols].mean()
                non_imputed_planted_avg = non_imputed_data[planted_cols].mean()
                imputed_harvested_avg = imputed_data[harvested_cols].mean()
                non_imputed_harvested_avg = non_imputed_data[harvested_cols].mean()

                # Calculate spread statistics
                imp_planted_stats = calculate_spread_stats(
                    imputed_planted_avg, "imputed_planted"
                )
                non_imp_planted_stats = calculate_spread_stats(
                    non_imputed_planted_avg, "non_imputed_planted"
                )
                imp_harvested_stats = calculate_spread_stats(
                    imputed_harvested_avg, "imputed_harvested"
                )
                non_imp_harvested_stats = calculate_spread_stats(
                    non_imputed_harvested_avg, "non_imputed_harvested"
                )

                print(f"\nPLANTED AREA SPREAD ANALYSIS:")
                print(
                    f'  Imputed:   {imp_planted_stats["spread_months"]:2d} active months, peak month {imp_planted_stats["peak_month"]:2d}, concentration {imp_planted_stats["concentration"]:.3f}, entropy {imp_planted_stats["entropy"]:.3f}'
                )
                print(
                    f'  Non-imputed: {non_imp_planted_stats["spread_months"]:2d} active months, peak month {non_imp_planted_stats["peak_month"]:2d}, concentration {non_imp_planted_stats["concentration"]:.3f}, entropy {non_imp_planted_stats["entropy"]:.3f}'
                )
                print(
                    f'  Active months - Imputed: {sorted(imp_planted_stats["active_months"])}'
                )
                print(
                    f'  Active months - Non-imputed: {sorted(non_imp_planted_stats["active_months"])}'
                )

                print(f"\nHARVESTED AREA SPREAD ANALYSIS:")
                print(
                    f'  Imputed:   {imp_harvested_stats["spread_months"]:2d} active months, peak month {imp_harvested_stats["peak_month"]:2d}, concentration {imp_harvested_stats["concentration"]:.3f}, entropy {imp_harvested_stats["entropy"]:.3f}'
                )
                print(
                    f'  Non-imputed: {non_imp_harvested_stats["spread_months"]:2d} active months, peak month {non_imp_harvested_stats["peak_month"]:2d}, concentration {non_imp_harvested_stats["concentration"]:.3f}, entropy {non_imp_harvested_stats["entropy"]:.3f}'
                )
                print(
                    f'  Active months - Imputed: {sorted(imp_harvested_stats["active_months"])}'
                )
                print(
                    f'  Active months - Non-imputed: {sorted(non_imp_harvested_stats["active_months"])}'
                )

                # Calculate spread similarity
                imp_active = set(imp_planted_stats["active_months"])
                non_imp_active = set(non_imp_planted_stats["active_months"])
                planted_spread_similarity = (
                    len(imp_active & non_imp_active) / len(imp_active | non_imp_active)
                    if len(imp_active | non_imp_active) > 0
                    else 0
                )

                imp_harv_active = set(imp_harvested_stats["active_months"])
                non_imp_harv_active = set(non_imp_harvested_stats["active_months"])
                harvested_spread_similarity = (
                    len(imp_harv_active & non_imp_harv_active)
                    / len(imp_harv_active | non_imp_harv_active)
                    if len(imp_harv_active | non_imp_harv_active) > 0
                    else 0
                )

                print(f"\nSPREAD SIMILARITY:")
                print(
                    f"  Planted area spread similarity: {planted_spread_similarity:.3f}"
                )
                print(
                    f"  Harvested area spread similarity: {harvested_spread_similarity:.3f}"
                )

                # Store results for comparison
                results[province] = {
                    "planted_similarity": planted_spread_similarity,
                    "harvested_similarity": harvested_spread_similarity,
                    "imp_planted_months": imp_planted_stats["spread_months"],
                    "non_imp_planted_months": non_imp_planted_stats["spread_months"],
                    "imp_harvested_months": imp_harvested_stats["spread_months"],
                    "non_imp_harvested_months": non_imp_harvested_stats[
                        "spread_months"
                    ],
                }

            elif imputed_mask.sum() > 0:
                print("Only imputed data available - no comparison possible")
            elif non_imputed_mask.sum() > 0:
                print("Only non-imputed data available - no comparison possible")
            else:
                print("No data available for analysis")
        else:
            print("No crop calendar data available")

    return results


def print_comparison_table(results_dict, setting_names):
    """Print a comparison table of results from different settings"""
    print(f"\n=== COMPARISON TABLE ===")
    print(f'{"Setting":<25} {"Buenos Aires":<15} {"Córdoba":<15} {"Santa Fe":<15}')
    print("-" * 70)

    for setting_name in setting_names:
        if setting_name in results_dict:
            results = results_dict[setting_name]
            ba_planted = results.get("Buenos Aires", {}).get("planted_similarity", 0)
            ba_harvested = results.get("Buenos Aires", {}).get(
                "harvested_similarity", 0
            )
            co_planted = results.get("Córdoba", {}).get("planted_similarity", 0)
            co_harvested = results.get("Córdoba", {}).get("harvested_similarity", 0)
            sf_planted = results.get("Santa Fe", {}).get("planted_similarity", 0)
            sf_harvested = results.get("Santa Fe", {}).get("harvested_similarity", 0)

            print(
                f"{setting_name:<25} {ba_planted:.3f},{ba_harvested:.3f}    {co_planted:.3f},{co_harvested:.3f}    {sf_planted:.3f},{sf_harvested:.3f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Test crop calendar imputation quality"
    )
    parser.add_argument(
        "--crop-calendar",
        default="data/argentina/final/crop_calendar_wheat.csv",
        help="Path to crop calendar CSV file",
    )
    parser.add_argument(
        "--yield-data",
        default="data/argentina/final/crop_wheat_yield_1970-2025.csv",
        help="Path to yield data CSV file",
    )
    parser.add_argument(
        "--setting-name",
        default="Current",
        help='Name for this setting (e.g., "3 neighbors + L2")',
    )

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.crop_calendar).exists():
        print(f"Error: Crop calendar file not found: {args.crop_calendar}")
        return 1

    if not Path(args.yield_data).exists():
        print(f"Error: Yield data file not found: {args.yield_data}")
        return 1

    # Run analysis
    results = analyze_imputation_quality(
        args.crop_calendar, args.yield_data, args.setting_name
    )

    print(f"\n=== SUMMARY FOR {args.setting_name.upper()} ===")
    for province, data in results.items():
        print(
            f'{province}: Planted={data["planted_similarity"]:.3f}, Harvested={data["harvested_similarity"]:.3f}'
        )

    return 0


if __name__ == "__main__":
    exit(main())
