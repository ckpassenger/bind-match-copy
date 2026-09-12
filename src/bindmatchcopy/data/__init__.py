"""Data utilities: prompt construction, IH filtering, streams, and data download."""

from .prompts import (
    SYNTH_NEEDLE_POOL,
    SYNTH_FILLER_POOL,
    QUERY_POS,
    make_pair,
    make_lag_sample,
)
from .ih_filter import (
    bigram_frequency_ratio,
    reliability_ratio,
    ih_composite_score,
    keep_paragraph,
    filter_paragraphs,
)
from .streams import IHPhase, IHPhaseSchedule, MixedIHStream
from .download import (
    download_openwebtext,
    download_wikitext103,
    download_code_python,
    download_math,
)
from .eval_data import (
    generate_fixed_evalset,
    load_fixed_evalset,
    DOMAIN_DIRS,
    DEFAULT_DOMAINS,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MAX_SEQS,
)

__all__ = [
    "SYNTH_NEEDLE_POOL",
    "SYNTH_FILLER_POOL",
    "QUERY_POS",
    "make_pair",
    "make_lag_sample",
    "bigram_frequency_ratio",
    "reliability_ratio",
    "ih_composite_score",
    "keep_paragraph",
    "filter_paragraphs",
    "IHPhase",
    "IHPhaseSchedule",
    "MixedIHStream",
    "download_openwebtext",
    "download_wikitext103",
    "download_code_python",
    "download_math",
    "generate_fixed_evalset",
    "load_fixed_evalset",
    "DOMAIN_DIRS",
    "DEFAULT_DOMAINS",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_MAX_SEQS",
]
