"""Sanity tests for poolsim. Run: python -m tests.test_poolsim  (from research/eli/src)

1. Renderer vs pyroomacoustics ShoeBox (order 1, frequency-flat walls): arrival
   times and relative amplitudes of the first arrivals agree.
2. Synthetic fit: recover perturbed hydrophone positions / sound speed / wall
   reflectivity from rendered early-window chirps at a few known poses.
3. Localize: fitted model + matched-field grid search finds a held-out pose.
"""
import math, sys, time
import numpy as np
import torch
import pyroomacoustics as pra
from poolsim import PoolRenderer, multires_stft_loss, localize, bandlimit

torch.manual_seed(0)
FS, NFFT = 96000, 8192
Lx, Ly, D, C = 25.0, 12.5, 2.0, 1480.0


def test_vs_pra():
    r = 0.8  # amplitude reflection, all walls incl. surface for this check
    absorption = 1 - r ** 2
    pra.constants.set("c", C)
    room = pra.ShoeBox([Lx, Ly, D], fs=FS, materials=pra.Material(absorption), max_order=1)
    src = [8.0, 4.0, 0.7]         # pra z measured from floor
    mic = [5.0, 6.0, 0.5]
    room.add_source(src); room.add_microphone(mic)
    room.compute_rir()
    rir = room.rir[0][0]

    m = PoolRenderer(Lx, Ly, D, C, n_hyd=1, order=1, fs=FS, n_fft=NFFT,
                     hyd_init=[[mic[0], mic[1], mic[2] - D]])
    m.surface_coef = r
    for w in m.walls.values():
        with torch.no_grad():
            w.a.fill_(torch.logit(torch.tensor(r)).item()); w.log_fc.fill_(math.log(1e7))
    h = m.cir(torch.tensor([src[0], src[1], src[2] - D]))[0].detach().numpy()

    # compare arrival times of the 7 strongest peaks
    def peaks(x, k=7):
        idx = np.argsort(np.abs(x))[::-1]
        out = []
        for i in idx:
            if all(abs(i - j) > 5 for j in out):
                out.append(i)
            if len(out) == k:
                break
        return np.sort(out)
    pp, hp = peaks(rir), peaks(h)
    # pra pads the RIR start by a fractional-delay filter half-length; align on the direct path
    off = pp[0] - hp[0]
    dt = np.abs((pp - off) - hp) / FS * 1e6
    amp_pra = np.abs(rir[pp]) / np.abs(rir[pp[0]])
    amp_our = np.abs(h[hp]) / np.abs(h[hp[0]])
    print(f"[vs pra] peak time error us: {dt.round(1)}  (max {dt.max():.1f})")
    print(f"[vs pra] rel amp pra: {amp_pra.round(3)}\n         rel amp our: {amp_our.round(3)}")
    assert dt.max() < 15, "arrival times disagree"
    assert np.abs(amp_pra - amp_our).max() < 0.1, "amplitudes disagree"


def make_truth():
    hyd = torch.tensor([[5.00, 6.00, -1.50], [5.06, 6.00, -1.50], [5.00, 6.06, -1.50], [5.06, 6.06, -1.50]])
    hyd = hyd + 0.01 * torch.randn(4, 3)
    t = PoolRenderer(Lx, Ly, D, 1495.0, order=1, fs=FS, n_fft=NFFT, hyd_init=hyd, wall_r0=0.85, floor_r0=0.6)
    with torch.no_grad():
        for k, w in t.walls.items():
            w.log_fc.fill_(math.log(6000.0 if k != "floor" else 3000.0))
    return t


def probe():
    dur, f0, f1 = 0.05, 2000.0, 18000.0
    tt = torch.arange(int(FS * dur)) / FS
    k = (f1 - f0) / dur
    x = torch.sin(2 * math.pi * (f0 * tt + 0.5 * k * tt * tt))
    return x


def test_fit_and_localize():
    truth = make_truth()
    p = probe()
    poses = torch.tensor([[8.0, 4.0, -1.0], [12.0, 8.0, -0.5], [3.0, 9.0, -1.2], [15.0, 3.0, -0.8], [7.0, 7.0, -1.6]])
    held = torch.tensor([10.0, 5.5, -1.1])
    max_delay = 0.03  # early window 30 ms
    with torch.no_grad():
        obs = [truth.render(q, p, max_delay) + 1e-3 * torch.randn(4, NFFT) for q in poses]
        obs_held = truth.render(held, p, max_delay)

    m = PoolRenderer(Lx, Ly, D, C, order=1, fs=FS, n_fft=NFFT)  # nominal init (6 cm square, c=1480, r=0.9/0.7)
    hyd_err0 = (m.hyd - truth.hyd).norm(dim=1).mean().item()
    opt = torch.optim.Adam([
        {"params": [m.hyd], "lr": 1e-3},
        {"params": [m.log_c], "lr": 1e-3},
        {"params": list(m.walls.parameters()) + [m.ch_gain, m.src_logmag_knots], "lr": 2e-2},
    ])  # ch_delay_us frozen: degenerate with hyd positions for a single fixed jig
    t0 = time.time()
    stages = [(0, 5000.0, 150), (0, 10000.0, 150), (0, 18000.0, 200)]  # coarse-to-fine band schedule
    it = 0
    for f_lo, f_hi, n in stages:
        obs_b = [bandlimit(o, FS, f_lo, f_hi) for o in obs]
        for _ in range(n):
            opt.zero_grad()
            loss = sum(multires_stft_loss(bandlimit(m.render(q, p, max_delay), FS, f_lo, f_hi), o) for q, o in zip(poses, obs_b)) / len(poses)
            loss.backward(); opt.step(); it += 1
            if it % 50 == 0:
                print(f"[fit] it {it} band<{f_hi:.0f} loss {loss.item():.4f} hyd err {(m.hyd - truth.hyd).norm(dim=1).mean().item()*100:.2f} cm c {m.c.item():.1f}")
    hyd_err = (m.hyd - truth.hyd).norm(dim=1).mean().item()
    print(f"[fit] {time.time()-t0:.1f}s; hyd err {hyd_err0*100:.2f} -> {hyd_err*100:.2f} cm; c {C} -> {m.c.item():.1f} (truth 1495)")
    for k in m.walls:
        print(f"[fit] wall {k}: r {torch.sigmoid(m.walls[k].a).item():.3f} fc {torch.exp(m.walls[k].log_fc).item():.0f}")

    # localize held-out pose with fitted model; obs spectrum deconvolved by probe
    P = torch.fft.rfft(p, n=NFFT)
    O = torch.fft.rfft(obs_held, n=NFFT) / (P[None] + 1e-6)
    m.eval()
    t0 = time.time()
    est, score = localize(m, O, bounds=((1, 24), (1, 11.5), (-1.9, -0.1)), max_delay=max_delay)
    err = (est - held).norm().item()
    print(f"[loc] est {est.numpy().round(2)} truth {held.numpy()} err {err*100:.1f} cm score {score:.3f} ({time.time()-t0:.1f}s CPU)")
    # baseline: nominal (unfitted) model
    m0 = PoolRenderer(Lx, Ly, D, C, order=1, fs=FS, n_fft=NFFT)
    est0, _ = localize(m0, O, bounds=((1, 24), (1, 11.5), (-1.9, -0.1)), max_delay=max_delay)
    print(f"[loc] nominal-model est {est0.numpy().round(2)} err {(est0-held).norm().item()*100:.1f} cm")


if __name__ == "__main__":
    test_vs_pra()
    test_fit_and_localize()
