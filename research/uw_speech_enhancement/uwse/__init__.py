"""Underwater speech enhancement: sim-to-real framework.

Subpackages
-----------
uwsim       pool channel simulator (signed image-source model, tail, low band,
            transducer chain, noise) and on-the-fly training-pair mixer
irtools     exponential-sine-sweep measurement and impulse-response analysis
evaluation  alignment, level normalization, ASR/WER, intrusive metrics, statistics
adapters    one SEAdapter per enhancement model (load, enhance, loss, params)
train       model-agnostic fine-tuning loop and datasets
bench       real-benchmark metadata schema, playlists and splits

See RESEARCH_PLAN.md for the specification each module implements.
"""

__version__ = "0.0.1"
