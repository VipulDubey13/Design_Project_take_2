from execution.experiment_runner import (
    build_cfg_from_c,
    run_comparative_study,
    run_failure_sweep
)
# CHANGE 1: Import ExecutionContext and CFGExecutionEngine so they are recognized here
from checkpointing.execution_context import ExecutionContext
from execution.cfg_execution_engine import CFGExecutionEngine

if __name__ == "__main__":

    # Path to your benchmark
    c_file_path = "sample_programs/crc.c"

    # --- THE "GOLDEN" PARAMETERS ---
    checkpoint_cost = 0.0001
    state_size_cost_factor = 0.0001

    # Failure rates: 1.0 (Stable) to 20.0 (Very Unstable/Stressed)
    stress_test_rates = [1.0, 5.0, 10.0, 20.0]

    # CHANGE 2: Add 'parser' to the variables receiving the return from build_cfg_from_c
    blocks, analysis, structural_metrics, parser = build_cfg_from_c(
        c_file_path,
        verbose=True
    )

    # 2. PHASE 2: DETAILED COMPARATIVE STUDY
    print("\n" + "="*40)
    print("DETAILED COMPARATIVE STUDY (λ = 5.0)")
    print("="*40)

    # CHANGE 3: We will compare all 3 strategies here for the professor
    strategies = ["ml_adaptive", "regression", "classification"]
    
    for strat in strategies:
        total_reads = 0
        total_writes = 0
        print(f"\nRunning 1000 iterations for {strat.upper()}...")
        
        for _ in range(1000):
            # Pass the strategy into the context
            context = ExecutionContext(
                failure_rate=5.0, 
                checkpoint_cost=checkpoint_cost, 
                structural_metrics=structural_metrics,
                strategy=strat
            )
            # Link the parser we got from Phase 1
            context.parser = parser 
            
            engine = CFGExecutionEngine(blocks, context)
            engine.execute()
            
            m = context.get_metrics()
            # Use .get() to avoid errors if metrics are empty
            total_reads += m.get('total_reads', 0)
            total_writes += m.get('total_writes', 0)

        print(f"[{strat.upper()}] Total Reads: {total_reads:,}")
        print(f"[{strat.upper()}] Total Writes: {total_writes:,}")

    # CHANGE 4: The standard study (Periodic vs Young's vs ML)
    # Ensure this function is updated in experiment_runner.py to handle the parser too
    run_comparative_study(
        blocks=blocks,
        structural_metrics=structural_metrics,
        failure_rate=5.0,
        checkpoint_cost=checkpoint_cost,
        parser=parser
    )

    # 3. PHASE 3: STOCHASTIC FAILURE SWEEP
    print("\n" + "="*40)
    print("STOCHASTIC FAILURE SWEEP (ML)")
    print("="*40)

    run_failure_sweep(
        blocks=blocks,
        structural_metrics=structural_metrics,
        failure_rates=stress_test_rates,
        trials_per_rate=20,
        parser=parser 
    )