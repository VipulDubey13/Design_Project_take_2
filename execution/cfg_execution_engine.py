import random
import time
from profiling.execution_profiler import ExecutionProfiler
from profiling.time_model import TimeModel

class CFGExecutionEngine:
    """
    The 'Heart' of the simulator.
    Steps through Basic Blocks and individual lines to simulate
    real-world hardware execution and checkpointing.
    Now enhanced for Triple-Pass Comparison (Periodic vs Young's vs ML).
    """

    def __init__(self, blocks, context):
        """
        blocks: dictionary {block_id: BasicBlock}
        context: The ExecutionContext (manages failures and saves)
        """
        self.blocks = blocks
        self.context = context
        self.current_block_id = 0

        # Profiling + time modeling
        self.profiler = ExecutionProfiler()
        self.time_model = TimeModel()

        # Expose profiler to context
        self.context.profiler = self.profiler

        # Track visit counts for state size estimation and loop detection
        self.visited_counts = {block_id: 0 for block_id in blocks.keys()}
        self.simulated_stack_depth = 0

    # --------------------------------------------------
    # SUCCESSOR SELECTION
    # --------------------------------------------------

    def choose_successor(self, block):
        if not block.successors:
            return None

        num_successors = len(block.successors)
        if num_successors == 1:
            return block.successors[0]

        # Dynamic weighting for branching (preserved from original)
        primary_weight = 0.7
        others_weight = 0.3 / (num_successors - 1)
        weights = [primary_weight] + [others_weight] * (num_successors - 1)

        return random.choices(block.successors, weights=weights, k=1)[0]

    # --------------------------------------------------
    # DYNAMIC STATE SIZE MODEL
    # --------------------------------------------------

    def compute_dynamic_state_size(self, block_id):
        unique_blocks_visited = sum(1 for count in self.visited_counts.values() if count > 0)
        loop_depth_factor = self.visited_counts[block_id]

        return (
                len(self.blocks)
                + unique_blocks_visited
                + loop_depth_factor
                + self.simulated_stack_depth
        )

    # --------------------------------------------------
    # EXECUTION LOOP (Triple-Pass Aware)
    # --------------------------------------------------

    def execute(self, max_steps=10000):
        """
        The main loop that steps through the program structure.
        Supports Periodic, Analytical (Young's), and ML Adaptive strategies.
        """
        # Determine verbosity: Only ML strategy shows line-level logs
        # This keeps the comparison table clean as per your request.
        verbose = (self.context.strategy == "ml_adaptive")
        steps = 0

        while steps < max_steps:
            block = self.blocks[self.current_block_id]
            self.visited_counts[self.current_block_id] += 1

            # --- 1. PROFILING THE BLOCK ---
            # Measure actual CPU time elapsed for this block
            start_time = time.perf_counter()

            # Simulated hardware latency (Preserved: 0.005s per block)
            time.sleep(0.005)

            end_time = time.perf_counter()
            measured_duration = end_time - start_time

            # --- 2. UPDATE TIME MODEL ---
            # Calibrate line-level timing based on the block measurement
            self.time_model.update_block_metrics(
                block_id=self.current_block_id,
                measured_time=measured_duration,
                lines=block.lines
            )

            # --- 3. LINE-BY-LINE EXECUTION ---
            for line_num, line_code in block.lines:
                # Track Memory Reads and Writes for this line
                reads, writes = self.context.parser.get_memory_ops(line_code)
                self.context.add_memory_ops(reads, writes)
                
                # Get instruction cost from the time model
                line_cost = self.time_model.get_line_cost(line_num)

                # IMPORTANT: Commit work to context.
                # This allows the simulator to trigger random failures based on λ.
                self.context.add_work(line_cost)

                # Evaluate Checkpoint decision based on current strategy (ML, Periodic, or Young's)
                dynamic_state_size = self.compute_dynamic_state_size(self.current_block_id)

                # Evaluate decision (The context handles the different strategy math)
                self.context.evaluate_checkpoint(
                    event_type=f"Line {line_num}: {line_code.strip()}",
                    state_size=dynamic_state_size,
                    current_line_cost=line_cost,
                    verbose=verbose # Only prints if it's the ML run
                )

            # --- 4. TRANSITION TO NEXT BLOCK ---
            next_block_id = self.choose_successor(block)

            if next_block_id is None:
                break

            # Heuristic for stack depth tracking (Preserved)
            last_line = block.lines[-1][1].strip()
            if "return" in last_line and self.simulated_stack_depth > 0:
                self.simulated_stack_depth -= 1
            elif "(" in last_line and ";" not in last_line:
                self.simulated_stack_depth += 1

            self.current_block_id = next_block_id
            steps += 1

        # Final Summary (Only for the ML run to avoid clutter)
        if verbose:
            final_time = self.time_model.get_total_execution_estimate()
            print(f"\n[Engine] {self.context.strategy.upper()} Pass Complete.")
            print(f"[Engine] Estimated Hardware Time: {final_time:.6f}s")