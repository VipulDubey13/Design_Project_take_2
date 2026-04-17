class FailureClassificationModel:
    """
    Classifies system state into: 0 (Safe), 1 (Risky - Checkpoint recommended)
    """
    def __init__(self):
        # Thresholds for a simple Decision Tree classifier
        self.work_threshold = 0.02
        self.lambda_threshold = 10.0

    def predict(self, work_since_last, failure_rate):
        # Decision Tree Logic
        if failure_rate > self.lambda_threshold:
            if work_since_last > (self.work_threshold / 2):
                return 1 # Checkpoint
        
        if work_since_last > self.work_threshold:
            return 1 # Checkpoint
            
        return 0 # Safe to continue

    def should_checkpoint(self, work_since_last, failure_rate):
        return self.predict(work_since_last, failure_rate) == 1