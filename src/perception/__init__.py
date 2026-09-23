"""Raw observation and model-context helpers."""

from src.perception.raw_observation import RawObservation


def __getattr__(name):
    # Payload readers are lower-level than ContextBuilder. Import the builder
    # lazily so memory modules can use those readers without a dependency cycle.
    if name == "ObservationContextBuilder":
        from src.perception.context_builder import ObservationContextBuilder
        return ObservationContextBuilder
    raise AttributeError(name)


__all__ = ["ObservationContextBuilder", "RawObservation"]
