"""Fast checks of the physical model and the validation target (no network needed)."""
import numpy as np
import pytest

from ai_siting import physics as P, sampling as S, config as C


def test_llama_405b_gpu_hours_within_10pct():
    # Meta Llama 3.1 model card: 30.84 M H100-hours for 3.8e25 FLOP
    assert abs(P.gpu_hours(3.8e25, S.central()) / 30.84e6 - 1) < 0.10


@pytest.mark.parametrize("arch", C.ARCHITECTURES)
def test_pue_above_one_and_rises_with_heat(arch):
    p = S.central()
    cold, _ = P.hourly_facility(np.array([0.0]), P.wet_bulb_stull([0.0], [70]), p, arch)
    hot, _ = P.hourly_facility(np.array([38.0]), P.wet_bulb_stull([38.0], [40]), p, arch)
    assert 1.0 < cold[0, 0] < hot[0, 0] < 1.6


def test_only_evaporative_design_uses_water():
    for arch in ("dry", "dtc_dry"):
        _, w = P.hourly_facility(np.array([35.0]), np.array([25.0]), S.central(), arch)
        assert np.all(w == 0)
    _, w = P.hourly_facility(np.array([35.0]), np.array([25.0]), S.central(), "evaporative")
    assert 1.4 < w[0, 0] < 2.5


def test_diesel_factor_matches_ipcc_default():
    assert P.diesel_ef_kg_per_kwh(0.35) == pytest.approx(0.762, abs=0.001)


def test_parameter_ranges_ordered():
    for k, v in C.PARAMS.items():
        assert v[1] <= v[2] <= v[3], k
