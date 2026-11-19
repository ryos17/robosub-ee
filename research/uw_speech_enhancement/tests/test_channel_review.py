"""Adversarial-review regression tests for uwse.uwsim.lowband / tail / physics.

Each test pins a defect found in review (RESEARCH_PLAN 4.2 rows 7, 8, 14).
"""

import time

import numpy as np
import pytest

from uwse.uwsim.lowband import lr4_split, modal_rir

FS = 8000


def _image_sum(src, rx, L, W, H, c, fs, N, T, per_image_scale):
    """Exact signed image sum (R_surface = -1 at z = 0, rigid walls and floor at z = H)."""
    p = np.arange(-15, 16)
    xs = np.concatenate([src[0] + 2 * p * L, -src[0] + 2 * p * L])
    ys = np.concatenate([src[1] + 2 * p * W, -src[1] + 2 * p * W])
    q = np.arange(-20, 21)[:, None] * 4 * H
    zz = np.concatenate([src[2] + q, -src[2] + q, 2 * H - src[2] + q, 2 * H + src[2] + q]).ravel()
    sz = np.concatenate([np.ones_like(q), -np.ones_like(q), np.ones_like(q), -np.ones_like(q)]).ravel()
    X, Y, Z = np.meshgrid(xs, ys, zz, indexing="ij")
    S = np.broadcast_to(sz, X.shape)
    d = np.sqrt((X - rx[0]) ** 2 + (Y - rx[1]) ** 2 + (Z - rx[2]) ** 2)
    keep = d / c < T + 0.01
    d, S = d[keep], S[keep]
    t = np.arange(N) / fs
    h = np.zeros(N)
    for i in range(0, len(d), 2048):
        h += (S[i : i + 2048] * per_image_scale(d[i : i + 2048])) @ np.sinc(fs * (t[None] - d[i : i + 2048, None] / c))
    return h


def _lp_rel_err(a, b, fs, f_lp=300.0, t0=0.01, t1=0.045):
    la, _ = lr4_split(a, fs, f_lp)
    lb, _ = lr4_split(b, fs, f_lp)
    sl = slice(int(t0 * fs), int(t1 * fs))
    return np.linalg.norm(la[sl] - lb[sl]) / np.linalg.norm(lb[sl]), np.linalg.norm(la[sl]) / np.linalg.norm(lb[sl])


GEOM = dict(src=(1.3, 2.1, 0.7), rx=(4.2, 3.3, 1.2), L=6.0, W=5.0, H=2.0, c=1500.0)


def test_modal_rir_default_level_matches_uwsim_ism_convention():
    """uwse.uwsim.ism writes A_i = R / r per image (no 4 pi).  render_rir splices
    modal_rir's low band into that ISM without a level fit, so modal_rir's default
    normalisation must be 1/r per image, not the Green's-function 1/(4 pi r)
    (which left the generic-pool low band 22 dB too quiet)."""
    g = GEOM
    T, N = 0.06, int(0.06 * FS)
    h_m = modal_rir(g["src"], g["rx"], g["L"], g["W"], g["H"], g["c"], FS, N, f_max=1500.0, damping=(1e6, 1e6))
    h_i = _image_sum(g["src"], g["rx"], g["L"], g["W"], g["H"], g["c"], FS, N, T, lambda d: 1.0 / d)
    err, ratio = _lp_rel_err(h_m, h_i, FS)
    assert ratio == pytest.approx(1.0, abs=0.02)
    assert err < 0.01
    # The Green's-function normalisation stays available.
    h_g = modal_rir(g["src"], g["rx"], g["L"], g["W"], g["H"], g["c"], FS, N, f_max=1500.0, damping=(1e6, 1e6),
                    spreading="4pi")
    np.testing.assert_allclose(h_g, h_m / (4 * np.pi), rtol=1e-9, atol=1e-12 * np.abs(h_m).max())


def test_modal_rir_matches_ism_module():
    """Cross-check against uwse.uwsim.ism (flat floor, R_s = -1, rigid walls/floor)."""
    try:
        from uwse.uwsim.geometry import Pool
        from uwse.uwsim.ism import ism_rir
    except Exception as exc:  # pragma: no cover - other module broken
        pytest.skip(f"uwse.uwsim.ism unavailable: {exc}")
    g = GEOM
    T, N = 0.06, int(0.06 * FS)
    res = ism_rir(np.array(g["src"]), np.array(g["rx"]), Pool(g["L"], g["W"], g["H"]), g["c"], FS, T,
                  surface_R=-1.0, walls=1.0, floor=1.0)
    h_i = np.zeros(N)
    h_i[: min(N, len(res.rir))] = res.rir[:N]
    h_m = modal_rir(g["src"], g["rx"], g["L"], g["W"], g["H"], g["c"], FS, N, f_max=1500.0, damping=(1e6, 1e6))
    err, ratio = _lp_rel_err(h_m, h_i, FS)
    assert ratio == pytest.approx(1.0, abs=0.03)
    assert err < 0.03


def test_modal_rir_decimated_equals_direct():
    """The decimated evaluation (internal rate >= oversample * f_max, then polyphase
    interpolation) must reproduce the direct evaluation at fs below f_max."""
    fs, n = 48000, 48000 // 4
    args = ((3.0, 2.0, 1.0), (9.0, 6.0, 2.5), 14.0, 9.0, 3.0, 1500.0, fs, n)
    kw = dict(f_max=600.0, damping=(0.6, 0.5))
    h_fast = modal_rir(*args, **kw)
    h_ref = modal_rir(*args, oversample=None, **kw)
    lf, _ = lr4_split(h_fast, fs, 400.0)
    lr, _ = lr4_split(h_ref, fs, 400.0)
    sl = slice(int(0.005 * fs), n - int(0.005 * fs))
    assert np.linalg.norm(lf[sl] - lr[sl]) / np.linalg.norm(lr[sl]) < 2e-3


def test_modal_rir_large_pool_is_fast():
    """Largest generic pool (50 x 25 x 5 m), f_max = 2 f_x = 1 kHz, 1 s at 48 kHz:
    ~8k modes.  The direct evaluation took ~14 s per RIR (8k x 48k sin+exp)."""
    t0 = time.perf_counter()
    h = modal_rir((10.0, 5.0, 2.0), (30.0, 12.0, 3.0), 50.0, 25.0, 5.0, 1500.0, 48000, 48000, f_max=1000.0)
    assert np.all(np.isfinite(h)) and h.shape == (48000,)
    assert time.perf_counter() - t0 < 8.0  # ~1.5-2 s now; generous for a loaded laptop


def test_splice_tail_per_band_level_continuity(rng):
    """Per-band (not just broadband) energy matching with a steep band spectrum.

    Reference: one continuous late_tail process.  Its first t_e is the "early" part;
    the spliced tail must continue every band at the reference's level.  Before the
    review fix the overlapping (+-0.25 oct), coherently-adding bands were not
    compensated and the valley bands came out ~2 dB high."""
    from uwse.uwsim.tail import late_tail, octave_filterbank, splice_tail

    fs, t_e = 16000, 0.04
    t60 = np.array([0.8, 0.6, 0.5, 0.4, 0.3, 0.25])
    lv = np.array([0.0, -12.0, 6.0, -15.0, 0.0, -20.0])
    n_e, length, R = int(t_e * fs), int(0.2 * fs), 30
    acc = acc_ref = 0.0
    for _ in range(R):
        full = late_tail(fs, length, t60, rng, levels_db=lv)
        out = splice_tail(full[:n_e], fs, t_e, t60, rng, length, match_window=0.03)
        acc = acc + octave_filterbank(out, fs) ** 2
        acc_ref = acc_ref + octave_filterbank(full, fs) ** 2
    i0, i1 = int(0.045 * fs), int(0.075 * fs)  # after the crossfade
    err_db = 10 * np.log10(acc[:, i0:i1].mean(1) / acc_ref[:, i0:i1].mean(1))
    # Residual: estimation noise of p_early over a 30 ms window (~10 dof at 500 Hz),
    # up to ~+0.9 dB in the valley band 1 over seeds 0-14; pre-fix bands 3, 5 >= 2.1 dB.
    assert np.all(np.abs(err_db) < 1.3), np.round(err_db, 2)
