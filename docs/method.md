# Method: Length and Dependence Modelling

This section summarizes the modelling choices behind CAPO and
EB–CAPO: how trajectory variance scales with length and dependence,
and how we use a simple Normal–Normal hierarchy to derive a tractable
Empirical Bayes objective.

## Length-only model and the ΔL family

Under an i.i.d. token model with

\[
  \mathrm{Var}(\boldsymbol{Z}_{i,t}) = \boldsymbol{\Sigma}_0,
  \qquad
  \mathrm{Cov}(\boldsymbol{Z}_{i,s}, \boldsymbol{Z}_{i,t}) = \boldsymbol{0}
  \text{ for } s \ne t,
\]

we have

\[
  \mathrm{Var}(\boldsymbol{g}_i) = L_i \boldsymbol{\Sigma}_0,
  \quad\Rightarrow\quad
  v_i \propto L_i.
\]

Plugging \(v_i \propto L_i\) into the MVU formula gives weights
proportional to \(L_i^{-1}\), which matches the “ΔL” rule from the
baseline paper (longer trajectories get downweighted linearly in their
length).

More generally, CAPO considers a **one-parameter family** of length
weights

\[
  x_i \;\propto\; L_i^{-\beta},
  \qquad \beta \in [0, 2],
\]

so that the ΔL rule corresponds to \(\beta = 1\), whereas
\(\beta < 1\) shifts mass towards longer trajectories, and
\(\beta > 1\) further downweights very long trajectories.

## Variance model with dependence

To handle correlated token-level noise, we introduce a parametric
variance model (one-dimensional version of §3):

\[
  g_i \mid \mu, \vartheta
    \sim \mathcal{N}\big(\mu, v_i(\vartheta)\big),
  \quad
  v_i(\vartheta)
    = \sigma^2 L_i^{\beta} s(L_i; \xi),
\]

with parameters \(\vartheta = (\sigma^2, \beta, \xi)\), where:

- \(\sigma^2\) – global scale,
- \(\beta\) – **length exponent** (recovers ΔL when \(\beta=1\)),
- \(s(L_i; \xi)\) – **dependence factor**, capturing how within-
  trajectory autocorrelation inflates or contracts variance beyond
  the i.i.d. \(L_i\) scaling.

It is convenient to define the **precision** (inverse variance) up to
a constant factor:

\[
  \omega_i(\beta, \xi)
    := L_i^{-\beta} s(L_i; \xi)^{-1}.
\]

The EB machinery will operate on these \(\omega_i\) rather than
\(\sigma^2\) directly.

## Empirical Bayes hierarchy

We put a flat prior on \(\mu\) and an inverse-gamma prior on
\(\sigma^2\):

\[
  p(\mu) \propto 1, \qquad
  \sigma^2 \sim \mathrm{Inv\text{-}Gamma}(a_0, b_0),
\]

while treating \((\beta, \xi)\) as hyperparameters with a
(separable) prior \(\pi_\beta(\beta)\pi_\xi(\xi)\).

Let

\[
  \lambda_i(\vartheta) = \frac{1}{v_i(\vartheta)}
    = \frac{\omega_i(\beta, \xi)}{\sigma^2},
\]

and define the sufficient statistics

\[
  T(\vartheta) = \sum_{i=1}^{G} \lambda_i(\vartheta) g_i,
  \qquad
  \Lambda(\vartheta) = \sum_{i=1}^{G} \lambda_i(\vartheta).
\]

The conditional posterior of \(\mu\) given \(\vartheta\) is Normal:

\[
  \mu \mid \vartheta, \{g_i\}
    \sim \mathcal{N}\!\big(m(\vartheta), \Lambda(\vartheta)^{-1}\big),
  \qquad
  m(\vartheta) = \frac{T(\vartheta)}{\Lambda(\vartheta)}.
\]

Factoring out \(\sigma^2\) leads to the EB quantities that appear
throughout the implementation:

- \(m(\beta,\xi)\) – precision-weighted mean,
- \(\omega_i(\beta,\xi) = L_i^{-\beta} s(L_i;\xi)^{-1}\),
- \(\Lambda_\omega(\beta,\xi) = \sum_i \omega_i(\beta,\xi)\),
- weighted residual sum of squares

  \[
    \mathrm{RSS}_\omega(\beta,\xi)
      = \sum_{i=1}^{G} \omega_i(\beta,\xi) \big(g_i - m(\beta,\xi)\big)^2.
  \]

These are exactly what `eb_stats` in the code computes.

## EB objective ℓ(β, ξ)

Integrating out \((\mu, \sigma^2)\) in the Normal–Inverse-Gamma model
leads to a marginal log posterior for \((\beta,\xi)\) of the form

\[
  \ell(\beta,\xi)
   = \log \pi_\beta(\beta) + \log \pi_\xi(\xi)
     + \frac{1}{2} \sum_i \log \omega_i(\beta,\xi)
     - \frac{1}{2} \log \Lambda_\omega(\beta,\xi)
     - \frac{G-1}{2} \log \mathrm{RSS}_\omega(\beta,\xi)
     + \text{const}.
\]

In practice we drop additive constants and set flat priors, so the
data-dependent terms dominate. The gradients implemented in
`grad_ell_beta` and `grad_ell_rho_eta` correspond to the derivations
in the paper:

- length direction (Corollary “grad‑beta”),
- dependence directions (Proposition “grad‑general” + Lemma
  “kband‑deriv”).
