import numpy as np

class FailureRegressionModel:
    """
    Predicts the 'Time to Next Failure' based on current work and lambda.
    Goal: Checkpoint if (Predicted_Time_Left < Line_Execution_Time).
    """
    def __init__(self):
        # Coefficients derived from training
        self.intercept = 0.05 
        self.coeff_work = -0.01
        self.coeff_lambda = -0.002

    def predict_time_to_failure(self, work_since_last, failure_rate):
        # Simple Linear Regression: Y = a + bX1 + cX2
        prediction = self.intercept + (self.coeff_work * work_since_last) + (self.coeff_lambda * failure_rate)
        return max(0, prediction)

    def should_checkpoint(self, work_since_last, failure_rate, next_line_cost):
        predicted_life = self.predict_time_to_failure(work_since_last, failure_rate)
        return predicted_life <= next_line_cost