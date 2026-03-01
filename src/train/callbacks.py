"""
Custom training callbacks.

- EvalLossDivergenceCallback: stop if eval loss spikes
- CostEstimatorCallback: print estimated $ spent
"""

from __future__ import annotations

import logging
import time

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EvalLossDivergenceCallback(TrainerCallback):
    """
    Stops training if eval loss exceeds *threshold* times the best eval loss
    seen so far, for *patience* consecutive evaluations.
    """

    def __init__(self, threshold: float = 1.5, patience: int = 3):
        self.threshold = threshold
        self.patience = patience
        self.best_eval_loss: float | None = None
        self.bad_evals = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.bad_evals = 0
            return

        if eval_loss > self.best_eval_loss * self.threshold:
            self.bad_evals += 1
            logger.warning(
                "Eval loss %.4f > %.1fx best %.4f (%d/%d patience)",
                eval_loss,
                self.threshold,
                self.best_eval_loss,
                self.bad_evals,
                self.patience,
            )
            if self.bad_evals >= self.patience:
                logger.error("Eval loss diverged — stopping training.")
                control.should_training_stop = True
        else:
            self.bad_evals = 0


class CostEstimatorCallback(TrainerCallback):
    """Prints estimated training cost at each logging step."""

    def __init__(self, gpu_hour_price: float = 1.10):
        self.gpu_hour_price = gpu_hour_price
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.start_time is None:
            return
        elapsed_hours = (time.time() - self.start_time) / 3600
        cost = elapsed_hours * self.gpu_hour_price
        logger.info(
            "Elapsed: %.2f hrs | Estimated cost: $%.2f (@ $%.2f/hr)",
            elapsed_hours,
            cost,
            self.gpu_hour_price,
        )
