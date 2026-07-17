"""tqp-reference-plugin: the reference out-of-tree quantizer for turboquant-pro."""

from .plugin import SPEC, ReferenceContainer, ReferenceUniformQuantizer

__all__ = ["SPEC", "ReferenceContainer", "ReferenceUniformQuantizer"]
__version__ = "0.1.0"
