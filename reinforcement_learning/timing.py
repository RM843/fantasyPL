class TimeTracker:
    """Tracks the execution time of various components in the training process."""
    def __init__(self):
        self.time_spent = {
            'action_selection': 0.0,
            'learning': 0.0,
            'plotting': 0.0,
            'environment_step': 0.0
        }

    def add_time(self, key: str, duration: float):
        """Add the duration to the specified key."""
        if key in self.time_spent:
            self.time_spent[key] += duration
        else:
            self.time_spent[key] = duration

    def print_time_breakdown(self):
        """Print a breakdown of the proportion of time spent on each part of the training process."""
        total_time = sum(self.time_spent.values())
        if total_time == 0:
            print("No time data to report.")
            return

        print("\nTraining Time Breakdown:")
        for key, value in self.time_spent.items():
            proportion = (value / total_time) * 100
            print(f"{key.capitalize()}: {proportion:.2f}% of training time")