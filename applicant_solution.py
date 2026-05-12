import json
from pathlib import Path

import gdown

import numpy as np
from scipy.io import loadmat

from task_and_baseline import CENTER, BW, baseline, build_task_helpers, make_bandpass

# Download the dataset
url = "https://drive.google.com/uc?id=1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4"
downloaded_file = "challenge.mat"
if not Path(downloaded_file).exists():
    gdown.download(url, downloaded_file, quiet=False)

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


RANK1_GAIN_SHRINK = 0.90
RANK1_GAIN_CANDIDATES = (0.75, 0.90, 1.00, 1.10, 1.20, 1.35)
CHANNEL_GAIN_SCALES = (0.75, 0.90, 1.00, 1.10, 1.20)
INVERSE_FILTER_REG = 1e-3
MIN_EXPLAIN_RATIO = 0.95
MAX_UNEXPLAINED_TO_RESIDUAL = 0.80


def _band_matrix(x):
    return np.column_stack([helpers["score_filter"](x[:, ch]) for ch in range(x.shape[1])])


def _rank1_from_band_matrix(band_matrix):
    cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = band_matrix @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    return np.column_stack(
        [
            (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
            for ch in range(band_matrix.shape[1])
        ]
    )


def _score_filter_response(n_samples):
    kernel = make_bandpass(CENTER, BW, Fs)
    padded = np.zeros(n_samples, dtype=np.complex128)
    padded[: len(kernel)] = kernel

    # In "same" convolution mode the kernel center is aligned with each sample.
    return np.fft.fft(np.roll(padded, -(len(kernel) // 2)))


def _undo_score_filter(band_component):
    response = _score_filter_response(band_component.shape[0])
    power = np.abs(response) ** 2
    inverse_response = response.conj() / (power + INVERSE_FILTER_REG * np.max(power))

    return np.column_stack(
        [
            np.fft.ifft(np.fft.fft(band_component[:, ch]) * inverse_response)
            for ch in range(band_component.shape[1])
        ]
    )


def _fit_rank1_gain(target_band, component):
    component_band = _band_matrix(component)
    denom = float(np.vdot(component_band, component_band).real) + 1e-30
    gain = float(np.vdot(component_band, target_band).real / denom)
    return RANK1_GAIN_SHRINK * float(np.clip(gain, 0.0, max(RANK1_GAIN_CANDIDATES)))


def _fit_rank1_channel_gains(target_band, component):
    component_band = _band_matrix(component)
    gains = []

    for ch in range(component.shape[1]):
        denom = np.vdot(component_band[:, ch], component_band[:, ch]) + 1e-30
        gain = np.vdot(component_band[:, ch], target_band[:, ch]) / denom
        max_gain = max(RANK1_GAIN_CANDIDATES)
        if np.abs(gain) > max_gain:
            gain *= max_gain / np.abs(gain)
        gains.append(RANK1_GAIN_SHRINK * gain)

    return np.asarray(gains, dtype=np.complex128)


def _is_valid_candidate(rx_before, rx_after):
    removed_band = _band_matrix(rx_before - rx_after)
    tx_part = helpers["fit_tx_prediction"](rx_before - rx_after)
    residual = removed_band - tx_part
    rank1_part = _rank1_from_band_matrix(residual)
    err = residual - rank1_part

    total_power = np.mean(np.abs(removed_band) ** 2) + 1e-30
    explain_ratio = 1.0 - np.mean(np.abs(err) ** 2) / total_power
    residual_band = _band_matrix(rx_after)
    err_powers = np.mean(np.abs(err) ** 2, axis=0)
    residual_powers = np.mean(np.abs(residual_band) ** 2, axis=0) + 1e-30

    return (
        explain_ratio >= MIN_EXPLAIN_RATIO
        and np.all(err_powers <= MAX_UNEXPLAINED_TO_RESIDUAL * residual_powers)
    )


def _average_reduction(rx_before_band, rx_after):
    rx_after_band = _band_matrix(rx_after)
    before_powers = np.mean(np.abs(rx_before_band) ** 2, axis=0) + 1e-30
    after_powers = np.mean(np.abs(rx_after_band) ** 2, axis=0) + 1e-30
    return float(np.mean(10.0 * np.log10(before_powers / after_powers)))


def _best_rank1_subtraction(rx_before, tx_prediction, rank1_component):
    rx_before_band = _band_matrix(rx_before)
    best_rx = rx_before - tx_prediction
    best_score = _average_reduction(rx_before_band, best_rx)

    residual_band = _band_matrix(best_rx)
    fitted_gain = _fit_rank1_gain(residual_band, rank1_component)
    channel_gains = _fit_rank1_channel_gains(residual_band, rank1_component)

    gains = sorted({0.0, fitted_gain, *(fitted_gain * scale for scale in RANK1_GAIN_CANDIDATES)})
    for gain in gains:
        candidate = rx_before - tx_prediction - gain * rank1_component
        if not _is_valid_candidate(rx_before, candidate):
            continue

        score = _average_reduction(rx_before_band, candidate)
        if score > best_score:
            best_score = score
            best_rx = candidate

    for scale in CHANNEL_GAIN_SCALES:
        candidate = rx_before - tx_prediction - rank1_component * (scale * channel_gains)[None, :]
        if not _is_valid_candidate(rx_before, candidate):
            continue

        score = _average_reduction(rx_before_band, candidate)
        if score > best_score:
            best_score = score
            best_rx = candidate

    return best_rx


def your_canceller(tx_n, rx):
    """Cancel TX-driven leakage, then remove the coherent external rank-1 term."""
    del tx_n

    tx_prediction = helpers["fit_tx_prediction"](rx)
    after_tx = rx - tx_prediction
    rank1_band = _rank1_from_band_matrix(_band_matrix(after_tx))
    rank1_prediction = _undo_score_filter(rank1_band)

    return _best_rank1_subtraction(rx, tx_prediction, rank1_prediction)


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
