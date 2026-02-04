# Algorithms: CAPO with Empirical Bayes

This page describes the three algorithmic components from §4, and
how they map to the functions in `capo/verl_integration/adv_estimators.py`.

## 1. EB–CAPO workflow (online)

The EB–CAPO workflow maintains streaming estimates
\((\beta_t, \xi_t)\) and uses them to reweight trajectories at each
outer step \(t\).

At step \(t\):

1. **Collect rollouts**

   Draw a batch

   \[
     \mathcal{B}_t
       = \{(g_i, L_i, \{Y_{i,\tau}\}_{\tau=1}^{L_i})\}_{i=1}^{G_t}
   \]

   under the current policy. Here:

   - \(g_i\) is a scalar (e.g. CAPO reward or gradient projection).
   - \(L_i\) is trajectory length.
   - \(Y_{i,\tau}\) are increments for ACF estimation (if used).

2. **E-step statistics (given \((\beta_{t-1}, \xi_{t-1})\))**

   For each trajectory:

   \[
     \omega_i^{(t)}
       \gets \big(L_i^{\beta_{t-1}}
                  s(L_i; \xi_{t-1})\big)^{-1}.
   \]

   Let \(Z_t = \sum_j \omega_j^{(t)}\) and normalized weights

   \[
     w_i^{(t)} = \omega_i^{(t)} / Z_t.
   \]

   The EB mean at step \(t\) is

   \[
     m_t = \sum_{i\in \mathcal{B}_t} w_i^{(t)} g_i.
   \]

   This is precisely what `eb_stats` computes in a single batch.

3. **EB updates**

   Using \((\omega_i^{(t)}, w_i^{(t)}, m_t, e_i^{(t)})\), form
   \(\Lambda_\omega^{(t)}\), \(\mathrm{RSS}_\omega^{(t)}\) and take a
   few gradient steps on \(\ell(\beta,\xi)\) with respect to β and ξ:

   - `grad_ell_beta` implements the length gradient direction
     (Corollary for ∂ℓ/∂β),
   - `grad_ell_rho_eta` implements dependence directions using
     the k-banded shape \(s(L;\rho,k,\eta)\) and Lemma on its
     derivatives.

   In the code, this is wrapped by
   `compute_capo_eb_full_advantage`, which performs a small number of
   gradient-ascent steps for β and ξ before computing final weights
   and advantages.

4. **Reweight and update policy**

   Use \(w_i^{(t)}\) as trajectory weights for CAPO (e.g. reweighting
   gradients or policy losses). In VERL, this corresponds to using
   the EB-derived trajectory advantages \(A_i\) inside PPO/GRPO-style
   updates.

### Mapping to code

- **Plain CAPO** (no EB):

  - Estimator: `capo`
  - Function: `compute_capo_advantage`
  - Algorithm: token-wise z-normalization over CAPO rewards.

- **EB–CAPO-lite** (length-only):

  - Algorithm: “EB-lite (no dependence)” in the text.
  - Function implementing EB fit:
    `eb_lite_fit_beta_and_weights(g, L)`
  - VERL estimator: `capo_eb_lite`
    (`compute_capo_eb_lite_advantage`):
    - Collapses to \((g_i, L_i)\),
    - Runs EB-lite,
    - Sets \(A_i = w_i (g_i - m)\),
    - Broadcasts \(A_i\) across tokens.

- **Full EB–CAPO** (length + dependence):

  - Dependence shape: `s_kband(L, rho, k, eta)`
    implements \(s(L; \rho, k, \eta)\).
  - EB statistics: `eb_stats(g, L, beta, rho, k, eta)` compute
    \(\omega_i\), \(w_i\), \(m\), \(e_i\),
    \(\Lambda_\omega\), \(\mathrm{RSS}_\omega\).
  - Gradients:
    - `grad_ell_beta` – ∂ℓ/∂β,
    - `grad_ell_rho_eta` – ∂ℓ/∂ρ, ∂ℓ/∂η.
  - ACF-moment:
    - `acf_moment_fit(Y, mask, k)` – moment-based estimator for
      \((\rho, \eta)\) from increments \(\{Y_{i,\tau}\}\).

  - VERL estimator: `capo_eb`
    (`compute_capo_eb_full_advantage`):
    - Optional ACF-based initialization of \((\rho, \eta)\),
    - Few gradient steps for β and ξ,
    - Final weights and advantages as in EB-lite.

---

## 2. EB-lite: log–log regression

EB-lite is a length-only, single-batch algorithm designed to sit
between fixed ΔL-style weights and the full EB–CAPO machinery.

Given \((g_i, L_i)\), it iteratively:

1. Computes residuals \(e_i = g_i - m^{(k)}\).
2. Forms

   \[
     z_i = \log(e_i^2 + \varepsilon).
   \]

3. Fits a linear regression

   \[
     z_i = c - \beta \log L_i + \varepsilon_i
   \]

   by least squares and sets \(\hat{\beta}^{(k+1)} = -\)slope.
4. Sets weights \(w_i \propto L_i^{-\hat{\beta}^{(k+1)}}\), normalizes
   them, and updates the aggregator

   \[
     m^{(k+1)} = \sum_i w_i g_i.
   \]

5. Repeats until \(m^{(k)}\) stabilizes.

The function `eb_lite_fit_beta_and_weights` is a direct translation
of this procedure, with a small number of outer iterations and a
tolerance on \(|m^{(k+1)} - m^{(k)}|\).

---

## 3. ACF-moment estimator

The ACF-moment estimator is used to initialize or refine the
dependence parameters ξ from token-level increments \(Y_{i,\tau}\).

Steps:

1. For each trajectory, compute empirical auto-covariances
   \(\hat{\gamma}_i(h)\) up to lag \(k\).
2. Aggregate to \(\bar{\gamma}(h)\), and normalize to obtain
   \(\bar{r}(h)\).
3. Fit the stretched-geometric model

   \[
     \bar{r}(h) \approx \rho^{h^{\eta}}
   \]

   by least squares over \(h = 1,\dots,k\).

In code, this is implemented as `acf_moment_fit`, which adopts a
coarse grid search over \(\rho\) and \(\eta\) and returns \((\hat{\rho}, \hat{\eta})\).
