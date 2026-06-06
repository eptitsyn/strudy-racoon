class DetectorError(Exception):
    """Base for all detector-module errors."""


class ModelNotRegisteredError(DetectorError):
    """Requested model name is not declared in the registry config."""


class ModelAlreadyLoadedError(DetectorError):
    """Attempt to load a model that is already loaded."""


class DetectorLoadError(DetectorError):
    """Underlying ML backend failed to load weights."""


class SwitchInProgressError(DetectorError):
    """Another switch is currently running and a new one cannot start."""
