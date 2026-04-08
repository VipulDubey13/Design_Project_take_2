from execution.experiment_runner import (
    build_cfg_from_c,
    run_comparative_study,
    run_failure_sweep
)

if __name__ == "__main__":

    # Path to your benchmark
    c_file_path = "sample_programs/crc.c"

    # --- THE "GOLDEN" PARAMETERS ---
    # 0.001 is a realistic cost for a lightweight memory save.
    # It allows the ML to be agile without "bankrupting" the efficiency.
    checkpoint_cost = 0.0001
    state_size_cost_factor = 0.0001

    # Failure rates: 1.0 (Stable) to 20.0 (Very Unstable/Stressed)
    stress_test_rates = [1.0, 5.0, 10.0, 20.0]

    # 1. PHASE 1: STATIC ANALYSIS
    # Builds the CFG and extracts structural weights
    blocks, analysis, structural_metrics = build_cfg_from_c(
        c_file_path,
        verbose=True
    )

    # 2. PHASE 2: DETAILED COMPARATIVE STUDY
    # We use λ = 5.0 here. This is high enough to cause failures,
    # but low enough that Periodic/Analytical won't just 'give up'.
    print("\n" + "="*40)
    print("DETAILED COMPARATIVE STUDY (λ = 5.0)")
    print("="*40)

    run_comparative_study(
        blocks=blocks,
        structural_metrics=structural_metrics,
        failure_rate=5.0,
        checkpoint_cost=checkpoint_cost
    )

    # 3. PHASE 3: STOCHASTIC FAILURE SWEEP (The Dataset Proof)
    # This proves the ML model's stability across different environments.
    print("\n" + "="*40)
    print("STOCHASTIC FAILURE SWEEP (ML)")
    print("="*40)

    run_failure_sweep(
        blocks=blocks,
        structural_metrics=structural_metrics,
        failure_rates=stress_test_rates,
        trials_per_rate=20
    )