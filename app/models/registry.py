"""Model registry — discoverable trainer and evaluator factories."""

from __future__ import annotations

from app.database.models import AlgorithmType


class TrainerRegistry:
    """Registry mapping AlgorithmType → trainer class."""

    _trainers: dict[AlgorithmType, type] = {}

    @classmethod
    def register(cls, algorithm: AlgorithmType):
        """Decorator to register a trainer class."""
        def decorator(trainer_cls):
            cls._trainers[algorithm] = trainer_cls
            return trainer_cls
        return decorator

    @classmethod
    def get_trainer(cls, algorithm: AlgorithmType):
        if algorithm not in cls._trainers:
            raise ValueError(f"No trainer registered for algorithm: {algorithm.value}")
        return cls._trainers[algorithm]()

    @classmethod
    def list_registered(cls) -> list[str]:
        return [a.value for a in cls._trainers.keys()]


class EvaluatorRegistry:
    """Registry mapping AlgorithmType → evaluator class."""

    _evaluators: dict[AlgorithmType, type] = {}

    @classmethod
    def register(cls, algorithm: AlgorithmType):
        def decorator(evaluator_cls):
            cls._evaluators[algorithm] = evaluator_cls
            return evaluator_cls
        return decorator

    @classmethod
    def get_evaluator(cls, algorithm: AlgorithmType):
        if algorithm not in cls._evaluators:
            raise ValueError(f"No evaluator registered for algorithm: {algorithm.value}")
        return cls._evaluators[algorithm]()

    @classmethod
    def list_registered(cls) -> list[str]:
        return [a.value for a in cls._evaluators.keys()]


# Convenience imports
def get_trainer(algorithm: AlgorithmType):
    return TrainerRegistry.get_trainer(algorithm)


def get_evaluator(algorithm: AlgorithmType):
    return EvaluatorRegistry.get_evaluator(algorithm)
