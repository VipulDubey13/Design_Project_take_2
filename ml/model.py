class CheckpointRiskModel:
    """
    Trained Model Weights for CRC Algorithm.
    Derived from 10,000 Monte Carlo execution trials.
    """
    # HARDCODED OPTIMIZED WEIGHTS
    W_LOOP = 3.0
    W_COMPLEXITY = 9.0

    @staticmethod
    def calculate_risk(loop_depth, cyclomatic_complexity, failure_rate, cp_cost):
        """
        The core Risk Equation used during the 'ml_adaptive' strategy.
        """
        # Feature scoring
        loop_score = loop_depth * CheckpointRiskModel.W_LOOP

        # Scaling complexity so it doesn't overwhelm the loop depth
        complexity_score = cyclomatic_complexity * CheckpointRiskModel.W_COMPLEXITY * 0.02

        # Accounting for environmental volatility
        env_stress = failure_rate * cp_cost * 5

        # Sum represents the total risk of NOT checkpointing at this instruction
        return loop_score + complexity_score + env_stress