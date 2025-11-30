# Problem Setup

## Observed quantities

We work at the trajectory level, in the same setting as PPO / GRPO /
VERL:

- A batch contains \(G\) trajectories \(i = 1,\dots,G\).
- Trajectory \(i\) has length \(L_i\) (number of response tokens).
- At each token \(t\) in trajectory \(i\) we have a contribution
  \(\boldsymbol{Z}_{i,t} \in \mathbb{R}^p\) (e.g. a score function
  term or gradient increment).

The unnormalized trajectory-level gradient is

\[
  \boldsymbol{g}_i \;=\; \sum_{t=1}^{L_i} \boldsymbol{Z}_{i,t}
  \;\in\; \mathbb{R}^p,
  \qquad
  \mathbb{E}[\boldsymbol{g}_i] = \boldsymbol{\mu} \in \mathbb{R}^p,
\]

which corresponds to equation \((\text{traj-grad})\) in the paper.
Within any batch, we assume the same underlying mean
\(\boldsymbol{\mu}\) for all trajectories.

In scalar form, for any unit direction \(\boldsymbol{u} \in \mathbb{R}^p\),

\[
  g_i := \boldsymbol{u}^\top \boldsymbol{g}_i, \qquad
  \mathbb{E}[g_i] = \mu := \boldsymbol{u}^\top \boldsymbol{\mu},
\]

and we denote the variance of this scalar by

\[
  v_i := \mathrm{Var}(g_i).
\]

The goal is to estimate \(\mu\) (or \(\boldsymbol{\mu}\)) as efficiently
as possible, and to use the same weights to construct stable policy
updates.

## Linear estimators and MVU weights

Consider estimators of the form

\[
  \hat{\mu}
    = \sum_{i=1}^{G} w_i g_i,
  \qquad
  \sum_{i=1}^{G} w_i = 1,
\]

which are unbiased as long as \(\mathbb{E}[g_i] = \mu\).

When the per-trajectory variances \(v_i\) are known, the classical
minimum-variance unbiased (MVU) weights are (equation
\((\text{mvu-weights})\)):

\[
  x_i^\star \;=\;
    \frac{1}{M}
    \frac{v_i^{-1}}{\sum_{j=1}^{G} v_j^{-1}},
\]

where \(M\) is the number of “effective components,” and the
normalization ensures \(\sum_i x_i^\star = 1\).

In other words, trajectories with smaller variance \(v_i\) receive
larger weight. CAPO focuses on **modelling and estimating** the
proportionality \(v_i \propto v(L_i, \text{dependence}_i)\) from the
observed batch, and then plugging this into the MVU form.
