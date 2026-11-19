"""uwsim: pool channel simulator and on-the-fly training-pair mixer (RESEARCH_PLAN.md section 4).

Sub-modules
-----------
physics, geometry, reflection, ism, tail, lowband   channel (section 4.2 rows 1-9)
chain        transducer/receive chain (rows 10-12, 19, 20)
noise        noise bank, bubbles, speech-band SNR mixing (rows 15-18, section 5.2)
config       SimConfig with every randomization range (section 4.2)
simulate     sample_scene / render_rir / simulate (section 4.1, targets section 8.2)
dataset      UWSimPairs (PyTorch) and RIRBank (section 5.3)

Imports are guarded so a missing or broken sibling module does not prevent importing
the rest of the package; ``IMPORT_ERRORS`` records what failed.
"""

from __future__ import annotations

import importlib

IMPORT_ERRORS: dict[str, str] = {}

_EXPORTS = {
    "config": ["SimConfig", "load_config", "save_config"],
    "chain": ["TransducerChain", "Projector", "Directivity", "Hydrophone", "drive_lpf",
              "hammerstein", "piston_directivity", "preamp_hpf", "adc", "bandlimit"],
    "noise": ["NoiseBank", "build_manifest", "bubble_noise", "mix_at_snr", "band_snr_db", "energy_vad"],
    "simulate": ["Scene", "sample_scene", "render_rir", "render_thruster_rir", "simulate", "build_chain",
                 "chain_delay", "direct_path_index"],
    "dataset": ["UWSimPairs", "RIRBank"],
    "geometry": ["Pool", "avery_competition_pool", "maas_dive_well"],
    "reflection": ["Boundary", "CONCRETE_UNCALIBRATED"],
    "ism": ["ism_rir", "ISMResult"],
    "tail": ["splice_tail"],
    "lowband": ["lr4_split", "splice_lowband", "modal_rir"],
    "physics": ["sound_speed_marczak", "octave_bands", "minnaert_frequency"],
}

__all__: list[str] = []
for _mod, _names in _EXPORTS.items():
    try:
        _m = importlib.import_module(f"{__name__}.{_mod}")
    except Exception as _e:  # pragma: no cover - depends on parallel work
        IMPORT_ERRORS[_mod] = f"{type(_e).__name__}: {_e}"
        continue
    for _n in _names:
        if hasattr(_m, _n):
            globals()[_n] = getattr(_m, _n)
            __all__.append(_n)
del _mod, _names
