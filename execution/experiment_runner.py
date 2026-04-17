import sys
import statistics
import random

from checkpointing.execution_context import ExecutionContext
from static_analysis.c_parser import CAlgorithmParser
from static_analysis.basic_block_builder import BasicBlockBuilder
from static_analysis.cfg_builder import CFGBuilder
from static_analysis.metrics_extractor import MetricsExtractor
from execution.cfg_execution_engine import CFGExecutionEngine


# ==========================================================
# STATIC ANALYSIS PIPELINE
# ==========================================================

def build_cfg_from_c(c_file_path: str, verbose=True):
    parser = CAlgorithmParser(c_file_path)
    parser.load()
    parser.analyze()
    program_representation = parser.get_program_representation()

    bb_builder = BasicBlockBuilder(program_representation)
    blocks = bb_builder.build()

    cfg_builder = CFGBuilder(blocks)
    blocks = cfg_builder.build()

    metrics_extractor = MetricsExtractor(blocks)
    structural_metrics = metrics_extractor.extract()

    if verbose:
        print("\n" + "=" * 40)
        print("PHASE 1: STATIC ANALYSIS")
        print("=" * 40)
        print(f"Algorithm: {parser.analyze()['algorithm'].upper()}")
        print("\nStructural Weights Extracted:")
        for k, v in structural_metrics.items():
            print(f"  {k}: {v}")

    # Ensure all four items are returned
    return blocks, structural_metrics, structural_metrics, parser


# ==========================================================
# COMPARATIVE STUDY
# ==========================================================

def run_comparative_study(blocks, structural_metrics, failure_rate, checkpoint_cost=0.01, parser=None):
    strategies = ["periodic", "analytical", "ml_adaptive", "regression", "classification"]
    results = {}
    study_seed = 42

    print("\n" + "=" * 40)
    print(f"PHASE 2: COMPARATIVE BENCHMARKING (λ = {failure_rate})")
    print("=" * 40)

    for strategy in strategies:
        context = ExecutionContext(
            failure_rate=failure_rate,
            checkpoint_cost=checkpoint_cost,
            state_size_cost_factor=0.001,
            structural_metrics=structural_metrics,
            strategy=strategy,
            seed=study_seed
        )

        context.parser = parser
        engine = CFGExecutionEngine(blocks, context)
        engine.execute()
        results[strategy] = context.get_metrics()

    print("\n" + "-" * 75)
    print(f"{'STRATEGY':<15} | {'CPs':<8} | {'FAILURES':<10} | {'RECOMPUTE':<12} | {'EFFICIENCY'}")
    print("-" * 75)

    for strategy, data in results.items():
        # Corrected Efficiency Formula: Work / (Work + Overhead + Recompute)
        # We assume total_work_time is data['total_time'] - data['recompute_time'] - (CPs * cost)
        work_time = max(0.0001, data.get('estimated_hardware_time', 0.03))
        total_lost = data['recompute_time'] + (data['checkpoint_count'] * checkpoint_cost)
        efficiency = (work_time / (work_time + total_lost)) * 100

        print(f"{strategy.upper():<15} | "
              f"{data['checkpoint_count']:<8} | "
              f"{data['failure_count']:<10} | "
              f"{data['recompute_time']:<12.4f} | "
              f"{min(99.99, efficiency):.2f}%")
    print("-" * 75)


# ==========================================================
# MULTI-TRIAL SWEEP
# ==========================================================

def run_trials(blocks, structural_metrics, failure_rate, checkpoint_cost=0.01, trials=20, parser=None):
    overheads = []

    for i in range(trials):
        context = ExecutionContext(
            failure_rate=failure_rate,
            checkpoint_cost=checkpoint_cost,
            state_size_cost_factor=0.001,
            structural_metrics=structural_metrics,
            strategy="ml_adaptive",
            seed=random.randint(0, 1000000)
        )

        context.parser = parser
        # KEY CHANGE: Strategy is "ml_adaptive" but we don't want it printing
        # the trace for 80 trials. We silence the engine here.
        engine = CFGExecutionEngine(blocks, context)

        # Capture stdout to silence the engine's print statements
        original_stdout = sys.stdout
        sys.stdout = None
        try:
            engine.execute()
        finally:
            sys.stdout = original_stdout

        metrics = context.get_metrics()
        overheads.append(metrics["overhead_ratio"])

    return {
        "mean_overhead": statistics.mean(overheads),
        "std_overhead": statistics.stdev(overheads) if trials > 1 else 0.0
    }


def run_failure_sweep(blocks, structural_metrics, failure_rates, trials_per_rate=20, parser=None ):
    print("\n" + "=" * 40)
    print("PHASE 3: STOCHASTIC FAILURE SWEEP (ML)")
    print("=" * 40)
    print(f"{'FAILURE RATE (λ)':<18} | {'AVG OVERHEAD':<15} | {'STDEV'}")
    print("-" * 55)

    for rate in failure_rates:
        summary = run_trials(blocks, structural_metrics, failure_rate=rate, trials=trials_per_rate, parser=parser)
        print(f"{rate:<18} | {summary['mean_overhead']:<15.4f} | {summary['std_overhead']:.4f}")

    print("-" * 55)