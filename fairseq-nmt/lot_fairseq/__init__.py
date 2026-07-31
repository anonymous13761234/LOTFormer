"""LOTFormer fairseq plug-in.

Use as a fairseq ``--user-dir``: pass ``--user-dir /path/to/fairseq-nmt/lot_fairseq``
and ``--arch lot_transformer_iwslt_de_en`` to fairseq-train / fairseq-generate.

The LOTFormer attention module (``LotMultiheadAttention``) is always importable; the
fairseq model registration (``lot_transformer``) is imported only if fairseq is present.
"""

from .lot_multihead_attention import LotMultiheadAttention  # noqa: F401

try:  # registering the architecture requires fairseq to be installed
    from . import lot_transformer  # noqa: F401
except Exception:  # pragma: no cover - fairseq not installed
    pass
