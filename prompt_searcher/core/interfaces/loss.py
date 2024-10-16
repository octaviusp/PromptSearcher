class LossFunction:
    def score(self, y_pred, y_true) -> int:
        """
        Calculate the loss score between predicted and true values.

        Args:
            y_pred: The predicted values.
            y_true: The true values.

        Returns:
            int: The calculated loss score.
        """
        raise NotImplementedError("This method should be overridden by subclasses")
