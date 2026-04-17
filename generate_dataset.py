import csv
import random
import tqdm  # Install this via 'pip install tqdm' for a progress bar
from execution.experiment_runner import build_cfg_from_c
from checkpointing.execution_context import ExecutionContext
from execution.cfg_execution_engine import CFGExecutionEngine


def generate_ml_dataset(c_file, output_file="ml_training_data.csv", samples=10000):
    # 1. Get the static structure once
    blocks, _, structural_metrics, parser = build_cfg_from_c(c_file, verbose=False)

    fieldnames = [
        'failure_rate', 'checkpoint_cost', 'loop_count',
        'cyclomatic_complexity', 'avg_block_size',
        'checkpoint_count', 'recompute_time', 'efficiency'
    ]

    print(f"Generating {samples} samples... this may take a few minutes.")

    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # tqdm provides a nice progress bar so you don't think it's frozen
        for _ in tqdm.tqdm(range(samples)):
            # Randomize environment for each run
            test_lambda = random.uniform(0.1, 100.0)
            test_cost = random.uniform(0.0001, 0.01)

            context = ExecutionContext(
                failure_rate=test_lambda,
                checkpoint_cost=test_cost,
                state_size_cost_factor=0.001,
                structural_metrics=structural_metrics,
                strategy="ml_adaptive",
                seed=random.randint(0, 10 ** 6)
            )

            context.parser = parser
            engine = CFGExecutionEngine(blocks, context)

            # Silent execution (no printing)
            engine.execute()
            metrics = context.get_metrics()

            # Save the result row
            writer.writerow({
                'failure_rate': test_lambda,
                'checkpoint_cost': test_cost,
                'loop_count': structural_metrics['loop_count'],
                'cyclomatic_complexity': structural_metrics['cyclomatic_complexity'],
                'avg_block_size': structural_metrics['average_block_size'],
                'checkpoint_count': metrics['checkpoint_count'],
                'recompute_time': metrics['recompute_time'],
                'efficiency': metrics['overhead_ratio']
            })


if __name__ == "__main__":
    generate_ml_dataset("sample_programs/crc.c")