"""
CAPO Test Suite.

This package contains unit and integration tests for the CAPO library.

Test Organization
-----------------
**Core EB Tests** (no VERL dependency):
    - `test_eb_toy.py`: 3-trajectory toy example from the paper
    - `test_eb_lite.py`: EB-lite recovery of β on synthetic data
    - `test_acf_moment.py`: ACF-moment recovery of ρ on AR(1) data

**VERL Integration Tests** (require VERL):
    - `test_adv_capo_base.py`: Plain CAPO advantage shape/masking
    - `test_eb_lite_and_full.py`: EB-lite and full EB advantage estimators
    - `test_s_kband_and_grad.py`: k-banded shape functions and gradients
    - `test_empirical_bayes_adv.py`: EB shrinkage behavior
    - `test_reward_fn.py`: CAPO reward function (outcome + process table)

Running Tests
-------------
From the repository root:

    # Run all tests
    pytest capo/tests/

    # Run only core tests (no VERL required)
    pytest capo/tests/test_eb_toy.py capo/tests/test_eb_lite.py capo/tests/test_acf_moment.py

    # Run with coverage
    pytest --cov=capo --cov-report=html capo/tests/

Test Philosophy
---------------
1. **Numerical sanity**: Tests verify algorithms produce finite, bounded outputs.
2. **Recovery tests**: Synthetic data tests verify estimators can recover
   ground-truth parameters (β, ρ, η) when the data generation matches the model.
3. **Shape tests**: VERL integration tests verify tensor shapes and masking.
4. **Invariance tests**: Tests verify expected mathematical properties
   (e.g., weight invariance to global length scaling).
"""
