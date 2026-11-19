"""Measurement and impulse-response analysis tools (RESEARCH_PLAN §4.3, §4.4, §6.4).

ess         exponential sine sweep, deconvolution, harmonic IRs / Hammerstein
            kernels, loopback clock-drift estimation and correction, sweep averaging
analysis    octave filterbank, Schroeder/Lundeby EDC, T20/T30/EDT, single- vs
            two-slope fits, DRR/C50, IR SNR, resolved arrivals, 1/3-octave coloration
soundspeed  TOF-slope sound speed and Marczak c(T) cross-check
"""

from .analysis import (
    OCTAVE_CENTERS,
    Arrivals,
    EDC,
    LundebyResult,
    SlopeFit,
    band_decay_times,
    band_edges,
    band_filter,
    c50_db,
    clarity_db,
    decay_time,
    decay_times,
    detect_arrivals,
    drr_db,
    filterbank,
    fit_decay_slopes,
    ir_snr_db,
    lundeby,
    onset_index,
    schroeder_edc,
    spectral_distance_db,
    third_octave_centers,
    third_octave_spectrum,
)
from .ess import (
    ESS,
    HarmonicIRs,
    SweepAverage,
    average_sweeps,
    correct_drift,
    deconvolve,
    estimate_drift_ppm,
    generate_ess,
    hammerstein_apply,
    hammerstein_kernels,
    hammerstein_matrix,
    harmonic_irs,
    harmonic_offsets,
    measure_ir,
    processing_gain_db,
)
from .soundspeed import (
    MarczakComparison,
    SoundSpeedFit,
    compare_marczak,
    fit_tof,
    marczak_c,
    marczak_dcdT,
    monte_carlo_tof_fit,
    predicted_rel_se_c,
    uniform_halfwidth_to_sd,
)
