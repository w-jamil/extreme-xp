#!/usr/bin/env python3
"""
algos.py — Complete collection of online learning algorithms in three families.

Family 1 — Online (Numba JIT accelerated):
    PA, Perceptron, OGC, AROW, RDA, SCW, AdaRDA
    Single-pass or multi-epoch on raw feature vectors.  Returns (y_pred, w).

Family 2 — Kernel Online (single-pass, growing support-vector set):
    KPA, KPerceptron, KOGC, KAROW, KRDA, KSCW, KAdaRDA
    Each sample may become a support vector.  Returns y_pred only.

Family 3 — Batch Kernel (classes with fit / predict / decision_function):
    KernelPA, KernelPerceptron, KernelGC, KernelAROW,
    KernelRDA, KernelSCW, KernelAdaRDA
    Budget-limited support vectors, multiple epochs, RBF kernel matrix.

Verified hyperparameter defaults
---------------------------------
PA          : parameter-free,
Perceptron  : parameter-free
OGC         : parameter-free,
AROW        : r = 0.1
RDA         : lambda_param = 1  (lambda=1 zeros weights on scaled data)
SCW         : eta = 0.9  → phi = norm.ppf(0.9) ≈ 1.28  (eta=0.5 → phi=0, degenerate)
AdaRDA      : lambda_param = AdaRDA
"""

import numpy as np
from numba import njit
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

# ═══════════════════════════════════════════════════════════════════════
#   SECTION 1 — RBF KERNEL FUNCTIONS                                   
# ══════════════════════════════════════════════════════════════════════

def rbf_kernel(x1, x2, gamma=1.0):
    """Compute the RBF (Gaussian) similarity between two single vectors.

    Measures how close x1 and x2 are: returns 1.0 when identical,
    decays towards 0.0 as they move apart.  The parameter gamma
    controls how fast the similarity drops — larger gamma means
    only very nearby points are considered similar.

    Parameters
    ----------
    x1 : ndarray (d,) — first feature vector
    x2 : ndarray (d,) — second feature vector
    gamma : float     — bandwidth (higher = tighter locality)

    Returns
    -------
    float — similarity score in (0, 1]
    """
    # Squared Euclidean distance between the two input vectors
    diff = x1 - x2
    # RBF (Gaussian) kernel value — always in (0, 1]
    return np.exp(-gamma * np.dot(diff, diff))


def rbf_kernel_matrix(X1, X2, gamma):
    """Build the full pairwise-similarity matrix between two sets of samples.

    Each entry K[i, j] tells you how similar sample i from the first set
    is to sample j from the second set, using the RBF (Gaussian) kernel.
    This is the core building block for all kernel algorithms: instead of
    working with raw features, they work with these similarity scores.

    The computation avoids an explicit double loop by expanding the
    squared distance as ||a - b||² = ||a||² + ||b||² - 2 a·b  and
    then applying exp(-gamma * distance²) element-wise.

    Parameters
    ----------
    X1 : ndarray (n1, d)   — first set of samples
    X2 : ndarray (n2, d)   — second set of samples
    gamma : float          — RBF bandwidth (higher = tighter locality)

    Returns
    -------
    K : ndarray (n1, n2)   — kernel Gram matrix, entries in (0, 1]
    """
    # ||X1[i]||² stored as column vector (n1, 1)
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    # ||X2[j]||² stored as column vector (n2, 1)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    # Cross-term matrix  X1 @ X2^T  of shape (n1, n2)
    cross = X1 @ X2.T
    # Squared distances via expansion; clamp negatives from floating-point error
    dist_sq = np.maximum(X1_sq + X2_sq.T - 2 * cross, 0.0)
    # Element-wise RBF
    return np.exp(-gamma * dist_sq)


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 2 — EVALUATION METRICS                                     
# ═══════════════════════════════════════════════════════════════════════

def calculate_class1_metrics(y_true, y_pred):
    """Score the model's predictions for the positive (minority) class.

    Returns five numbers that tell you how well the model detects
    the "interesting" class (e.g. fraud, attack, disease):

    - Precision — of all the samples the model *called* positive,
                   what fraction actually was?  (Low = too many false alarms)
    - Recall    — of all the truly positive samples, what fraction did
                   the model catch?  (Low = missing real positives)
    - FNR       — false negative rate: 1 − Recall (fraction missed)
    - FPR       — false positive rate: fraction of negatives wrongly flagged
    - F1        — harmonic mean of Precision and Recall; single summary score

    Parameters
    ----------
    y_true : array-like — ground-truth labels
    y_pred : array-like — predicted labels

    Returns
    -------
    precision, recall, fnr, fpr, f1 : floats
    """
    try:
        # Collect every label that appears in either the true or predicted arrays
        unique_labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))

        if len(unique_labels) <= 2:
            # --- binary classification ---
            labels = sorted(unique_labels)
            # Heuristic: the positive class is the one that looks "positive"
            positive_indicators = {1, '1', 'fraud', 'malicious', 'attack', 'positive', 'anomaly'}
            pos_label = None
            for lbl in labels:
                if lbl in positive_indicators or (isinstance(lbl, (int, float)) and lbl > 0):
                    pos_label = lbl
                    break
            # Fallback: second label in sorted order is positive
            if pos_label is None:
                pos_label = labels[1] if len(labels) > 1 else labels[0]
            # Negative label is whichever is not positive
            neg_label = labels[0] if labels[0] != pos_label else (labels[1] if len(labels) > 1 else labels[0])
            labels = [neg_label, pos_label]
        else:
            labels = sorted(unique_labels)

        # Build the confusion matrix with the chosen label order
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Multi-class fallback: first class is negative, rest are positive
            tn = cm[0, 0]
            fp = cm[0, 1:].sum()
            fn = cm[1:, 0].sum()
            tp = cm[1:, 1:].sum()
    except (ValueError, IndexError) as e:
        print(f"Error in calculate_class1_metrics: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall = TP / (TP + FN)   (also called TPR / sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # False Negative Rate = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    # False Positive Rate = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # F1 = harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, fnr, fpr, f1


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 3 — NUMBA PREDICTION UTILITY                              
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _predict_val(X_val, w):
    """Fast validation prediction using Numba.

    Computes sign(w · x) for every sample in X_val.  Ties (exactly zero)
    default to +1.  Runs as compiled machine code for speed.

    Parameters
    ----------
    X_val : ndarray (n, d) — validation features (contiguous float64)
    w     : ndarray (d,)   — weight vector (contiguous float64)

    Returns
    -------
    y_pred : ndarray (n,) — predicted labels in {-1, +1}
    """
    n = X_val.shape[0]
    d = X_val.shape[1]
    y_pred = np.zeros(n)
    for i in range(n):
        # Dot product  w^T x_i
        s = 0.0
        for j in range(d):
            s += X_val[i, j] * w[j]
        # Sign classification with +1 default on exact zero
        if s > 0:
            y_pred[i] = 1.0
        elif s < 0:
            y_pred[i] = -1.0
        else:
            y_pred[i] = 1.0
    return y_pred


def predict_val(X_val, w):
    """Predict labels for a validation set using a learned weight vector.

    Thin wrapper that converts inputs to the right format (contiguous
    float64 arrays) and then calls the fast Numba-compiled predictor.

    Parameters
    ----------
    X_val : array-like (n, d) — validation features
    w     : array-like (d,)   — weight vector

    Returns
    -------
    y_pred : ndarray (n,) — predicted labels in {-1, +1}
    """
    return _predict_val(
        np.ascontiguousarray(X_val, dtype=np.float64),
        np.ascontiguousarray(w,     dtype=np.float64),
    )


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 4 — NUMBA JIT INNER-LOOP KERNELS                           
#                                                                       
#   Each kernel processes one full epoch of permuted samples.            
#   State arrays (w, Sigma, g, …) are updated IN-PLACE for speed.       
# ═══════════════════════════════════════════════════════════════════════

# ---------- 4a. Passive-Aggressive (PA) ----------
@njit(cache=True)
def _pa_kernel(X, y, perm, w):
    """Passive-Aggressive inner loop — one epoch of online updates.

    Idea: when the model makes a mistake (or is not confident enough),
    it takes the smallest weight adjustment that would fix that mistake.
    The step size is loss / ||x||⁴ — bigger corrections for clearer
    errors, smaller ones for borderline cases.  If the prediction is
    already correct and confident, the weights stay unchanged.

    Parameters
    ----------
    X    : ndarray (n, d)  — training features
    y    : ndarray (n,)    — true labels in {-1, +1}
    perm : ndarray (n,)    — index permutation (shuffle order)
    w    : ndarray (d,)    — weight vector, updated in-place

    Returns
    -------
    y_pred : ndarray (n,) — predictions made *before* each update
    w      : ndarray (d,) — updated weight vector
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Raw score  w · x
        score = 0.0
        for j in range(n_features):
            score += x[j] * w[j]
        # Discrete prediction via sign
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
        y_pred[i] = pred
        # Hinge loss on discrete prediction (not continuous margin)
        loss = 1.0 - y_actual * pred
        if loss < 0.0:
            loss = 0.0
        # PA update only when loss > 0
        if loss > 0.0:
            # Squared L2 norm of current sample
            l2sq = 0.0
            for j in range(n_features):
                l2sq += x[j] * x[j]
            if l2sq > 0.0:
                l2sq_sq = l2sq * l2sq
                eta = loss / l2sq_sq
                for j in range(n_features):
                    w[j] += eta * y_actual * x[j]
    return y_pred, w


# ---------- 4b. Perceptron ----------
@njit(cache=True)
def _percept_kernel(X, y, perm, w):
    """Perceptron inner loop — the simplest online classifier.

    Idea: when the model's prediction disagrees with the true label,
    add (or subtract) the input vector to the weights.  Over time,
    the weight vector tilts toward the correct separating direction.
    No update at all if the prediction is already correct.

    Parameters
    ----------
    X    : ndarray (n, d)  — training features
    y    : ndarray (n,)    — true labels in {-1, +1}
    perm : ndarray (n,)    — index permutation (shuffle order)
    w    : ndarray (d,)    — weight vector, updated in-place

    Returns
    -------
    y_pred : ndarray (n,) — predictions made before each update
    w      : ndarray (d,) — updated weight vector
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Dot product w · x
        score = 0.0
        for j in range(n_features):
            score += x[j] * w[j]
        # Discrete prediction
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
        y_pred[i] = pred
        # Additive update on mis-classification
        if pred != y_actual:
            for j in range(n_features):
                w[j] += y_actual * x[j]
    return y_pred, w


# ---------- 4c. OGC (Online Gradient Classification) ----------
@njit(cache=True)
def _ogc_kernel(X, y, perm, w):
    """Online Gradient Classification inner loop — gradient step on every sample.

    Idea: unlike Perceptron (which only updates on errors), OGC nudges
    the weights on *every* sample, proportional to (true − predicted)
    normalised by the length of the feature vector.  This makes it
    more responsive to subtle patterns but also noisier.

    Parameters
    ----------
    X    : ndarray (n, d)  — training features
    y    : ndarray (n,)    — true labels in {-1, +1}
    perm : ndarray (n,)    — index permutation (shuffle order)
    w    : ndarray (d,)    — weight vector, updated in-place

    Returns
    -------
    y_pred : ndarray (n,) — predictions made before each update
    w      : ndarray (d,) — updated weight vector
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_act = y[idx]
        # Compute prediction score
        score = 0.0
        for j in range(n_features):
            score += x[j] * w[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
        y_pred[i] = pred
        # L2 norm of current feature vector
        xnorm = 0.0
        for j in range(n_features):
            xnorm += x[j] * x[j]
        # Normalised gradient step (applied every sample, not just errors)
        scale = (y_act - pred) / (np.sqrt(xnorm) + 1e-8)
        for j in range(n_features):
            w[j] += scale * x[j]
    return y_pred, w


# ---------- 4d. AROW — diagonal covariance ----------
@njit(cache=True)
def _arow_diag_kernel(X, y, perm, u, sigma_diag, r):
    """AROW (diagonal) inner loop — confidence-aware online learning.

    Idea: maintain a per-feature "uncertainty" score (sigma_diag).
    Features the model is unsure about get larger updates; features
    it is already confident about get smaller ones.  This prevents
    over-reacting to noise on well-learned features while staying
    aggressive on under-learned ones.  The parameter r controls
    overall caution — larger r = more conservative updates.

    Uses a diagonal approximation: each feature's uncertainty is
    tracked independently (O(d) instead of O(d*d) for full matrix).

    Parameters
    ----------
    X          : ndarray (n, d) — training features
    y          : ndarray (n,)   — true labels in {-1, +1}
    perm       : ndarray (n,)   — index permutation
    u          : ndarray (d,)   — mean weight vector, updated in-place
    sigma_diag : ndarray (d,)   — per-feature uncertainty, updated in-place
    r          : float          — regularisation (higher = more cautious)

    Returns
    -------
    y_pred     : ndarray (n,)  — predictions before each update
    u          : ndarray (d,)  — updated weight vector
    sigma_diag : ndarray (d,)  — updated uncertainty scores
    """
    n_samples = X.shape[0]
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Prediction: u · x
        score = 0.0
        for j in range(x.shape[0]):
            score += x[j] * u[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 1.0)
        y_pred[i] = pred
        # Hinge loss
        margin = y_actual * score
        lt = 1.0 - margin
        if lt < 0.0:
            lt = 0.0
        # Confidence v_t = x^T diag(σ) x
        vt = 0.0
        for j in range(x.shape[0]):
            vt += x[j] * x[j] * sigma_diag[j]
        if lt > 0.0:
            denom = vt + r
            if denom > 0.0:
                alpha_t = lt / denom
                beta_t = 1.0 / denom
            else:
                alpha_t = 0.0
                beta_t = 0.0
            # Mean update: u ← u + α y Σ x  (diagonal Σ)
            for j in range(x.shape[0]):
                u[j] += alpha_t * y_actual * sigma_diag[j] * x[j]
            # Covariance shrink: σ_j ← σ_j − β σ_j² x_j²
            for j in range(x.shape[0]):
                sigma_diag[j] -= beta_t * sigma_diag[j] * sigma_diag[j] * x[j] * x[j]
                # Floor to prevent numerical collapse to zero
                if sigma_diag[j] < 1e-10:
                    sigma_diag[j] = 1e-10
    return y_pred, u, sigma_diag


# ---------- 4e. AROW — full covariance ----------
@njit(cache=True)
def _arow_full_kernel(X, y, perm, u, Sigma, r):
    """AROW (full covariance) inner loop — tracks feature correlations.

    Same idea as diagonal AROW, but instead of treating each feature's
    uncertainty independently, this version keeps a full d by d matrix
    that also captures how features relate to each other.  More
    accurate when features are correlated, but costs O(d*d) per update.

    Parameters
    ----------
    X     : ndarray (n, d)    — training features
    y     : ndarray (n,)      — true labels in {-1, +1}
    perm  : ndarray (n,)      — index permutation
    u     : ndarray (d,)      — mean weight vector, updated in-place
    Sigma : ndarray (d, d)    — covariance (uncertainty) matrix, updated in-place
    r     : float             — regularisation (higher = more cautious)

    Returns
    -------
    y_pred : ndarray (n,)    — predictions before each update
    u      : ndarray (d,)    — updated weight vector
    Sigma  : ndarray (d, d)  — updated covariance matrix
    """
    n_samples, n_features = X.shape
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Prediction
        score = 0.0
        for j in range(n_features):
            score += x[j] * u[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 1.0)
        y_pred[i] = pred
        # Hinge loss
        margin = y_actual * score
        lt = 1.0 - margin
        if lt < 0.0:
            lt = 0.0
        # Σ x  (matrix-vector product)
        Sigma_x = np.zeros(n_features)
        for j in range(n_features):
            for k in range(n_features):
                Sigma_x[j] += Sigma[j, k] * x[k]
        # v_t = x^T Σ x = x · Σ_x
        vt = 0.0
        for j in range(n_features):
            vt += x[j] * Sigma_x[j]
        if lt > 0.0:
            denom = vt + r
            if denom > 0.0:
                alpha_t = lt / denom
                beta_t = 1.0 / denom
            else:
                alpha_t = 0.0
                beta_t = 0.0
            # Mean update:  u ← u + α y Σ_x
            for j in range(n_features):
                u[j] += alpha_t * y_actual * Sigma_x[j]
            # Covariance shrink:  Σ ← Σ − β (Σ_x)(Σ_x)^T
            for j in range(n_features):
                for k in range(n_features):
                    Sigma[j, k] -= beta_t * Sigma_x[j] * Sigma_x[k]
    return y_pred, u, Sigma


# ---------- 4f. RDA (Regularised Dual Averaging) ----------
@njit(cache=True)
def _rda_kernel(X, y, perm, w, g, lambda_param, gamma_param, epoch, n_samples_total):
    """RDA inner loop — builds sparse weight vectors by averaging gradients.

    Idea: instead of applying each gradient immediately, RDA keeps a
    running average of all past gradients.  Weights are then reconstructed
    by soft-thresholding that average at the level lambda.  Features
    whose average gradient magnitude stays below lambda are forced to
    exactly zero — producing genuinely sparse models where irrelevant
    features are automatically switched off.

    Parameters
    ----------
    X              : ndarray (n, d)  — training features
    y              : ndarray (n,)    — true labels in {-1, +1}
    perm           : ndarray (n,)    — index permutation
    w              : ndarray (d,)    — weight vector, reconstructed each step
    g              : ndarray (d,)    — running average gradient, updated in-place
    lambda_param   : float           — sparsity threshold (higher = more zeros)
    gamma_param    : float           — step-size denominator
    epoch          : int             — current epoch number (for global time step)
    n_samples_total: int             — samples per epoch

    Returns
    -------
    y_pred : ndarray (n,) — predictions before each update
    w      : ndarray (d,) — reconstructed weight vector
    g      : ndarray (d,) — updated gradient average
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        # Global time step across all epochs
        t = i + 1 + epoch * n_samples_total
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Prediction
        score = 0.0
        for j in range(n_features):
            score += x[j] * w[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 1.0)
        y_pred[i] = pred
        # Hinge loss
        margin = y_actual * score
        lt = 1.0 - margin
        if lt < 0.0:
            lt = 0.0
        # Update gradient running average
        if lt > 0.0:
            for j in range(n_features):
                # Sub-gradient of hinge loss: −y x
                gt_j = -y_actual * x[j]
                g[j] = ((t - 1) / t) * g[j] + (1.0 / t) * gt_j
        else:
            # No loss ⇒ zero sub-gradient, still decay the average
            for j in range(n_features):
                g[j] = ((t - 1) / t) * g[j]
        # Reconstruct weights via soft-thresholding at λ
        sqrt_t = np.sqrt(float(t))
        for j in range(n_features):
            if np.abs(g[j]) > lambda_param:
                # RDA weight: −(√t / γ)(ḡ_j − λ sign(ḡ_j))
                w[j] = -(sqrt_t / gamma_param) * (g[j] - lambda_param * np.sign(g[j]))
            else:
                # Below threshold ⇒ zero (sparsity)
                w[j] = 0.0
    return y_pred, w, g


# ---------- 4g. SCW — diagonal Σ ----------
@njit(cache=True)
def _scw_diag_kernel(X, y, perm, u, sigma_diag, C, phi):
    """SCW-I (diagonal) inner loop — confidence-weighted with soft margin.

    Idea: like AROW, SCW tracks per-feature uncertainty.  The key
    difference is the loss function: SCW triggers an update whenever
    the model's *confidence* (not just its prediction) is too low.
    Even a correct prediction will be updated if the model was unsure.
    The parameter phi (derived from a confidence level like 90%)
    sets how demanding the confidence requirement is, and C caps
    the maximum step size to prevent over-correction on outliers.

    Parameters
    ----------
    X          : ndarray (n, d) — training features
    y          : ndarray (n,)   — true labels in {-1, +1}
    perm       : ndarray (n,)   — index permutation
    u          : ndarray (d,)   — mean weight vector, updated in-place
    sigma_diag : ndarray (d,)   — per-feature uncertainty, updated in-place
    C          : float          — aggressiveness cap (max step size)
    phi        : float          — confidence demand (from probit of eta)

    Returns
    -------
    y_pred     : ndarray (n,)  — predictions before each update
    u          : ndarray (d,)  — updated weight vector
    sigma_diag : ndarray (d,)  — updated uncertainty scores
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.zeros(n_samples)
    phi_sq = phi * phi
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Prediction
        score = 0.0
        for j in range(n_features):
            score += x[j] * u[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 1.0)
        y_pred[i] = pred
        # Diagonal confidence  v_t = Σ σ_j x_j²
        vt = 0.0
        for j in range(n_features):
            vt += x[j] * x[j] * sigma_diag[j]
        # Signed margin
        mt = y_actual * score
        # SCW loss = φ √v_t − m_t  (clamped to ≥ 0)
        lt = phi * np.sqrt(vt) - mt
        if lt < 0.0:
            lt = 0.0
        if lt > 0.0:
            # SCW-I closed-form step size
            pa = 1.0 + phi_sq / 2.0
            xi = 1.0 + phi_sq
            sqrt_term = (mt * mt * phi_sq * phi_sq / 4.0) + (vt * phi_sq * xi)
            if sqrt_term < 0.0:
                sqrt_term = 0.0
            alpha_t = (1.0 / (vt * xi)) * (-mt * pa + np.sqrt(sqrt_term))
            # Clamp α to [0, C]
            if alpha_t < 0.0:
                alpha_t = 0.0
            if alpha_t > C:
                alpha_t = C
            # Covariance shrink factor
            sqrt_ut_term = (alpha_t * alpha_t * vt * vt * phi_sq) + (4.0 * vt)
            if sqrt_ut_term < 0.0:
                sqrt_ut_term = 0.0
            ut_val = 0.25 * (-alpha_t * vt * phi + np.sqrt(sqrt_ut_term)) ** 2
            beta_t = (alpha_t * phi) / (np.sqrt(ut_val) + vt * alpha_t * phi + 1e-8)
            # Mean update
            for j in range(n_features):
                u[j] += alpha_t * y_actual * sigma_diag[j] * x[j]
            # Variance shrink
            for j in range(n_features):
                sigma_diag[j] -= beta_t * sigma_diag[j] * sigma_diag[j] * x[j] * x[j]
                if sigma_diag[j] < 1e-10:
                    sigma_diag[j] = 1e-10
    return y_pred, u, sigma_diag


# ---------- 4h. SCW — full Σ ----------
@njit(cache=True)
def _scw_full_kernel(X, y, perm, u, Sigma, C, phi):
    """SCW-I (full covariance) inner loop — tracks feature correlations.

    Same confidence-weighted approach as the diagonal version, but
    maintains a full d by d covariance matrix to capture how features
    relate.  More expressive but costs O(d²) per sample.

    Parameters
    ----------
    X     : ndarray (n, d)    — training features
    y     : ndarray (n,)      — true labels in {-1, +1}
    perm  : ndarray (n,)      — index permutation
    u     : ndarray (d,)      — mean weight vector, updated in-place
    Sigma : ndarray (d, d)    — covariance matrix, updated in-place
    C     : float             — aggressiveness cap
    phi   : float             — confidence demand

    Returns
    -------
    y_pred : ndarray (n,)    — predictions before each update
    u      : ndarray (d,)    — updated weight vector
    Sigma  : ndarray (d, d)  — updated covariance matrix
    """
    n_samples, n_features = X.shape
    y_pred = np.zeros(n_samples)
    phi_sq = phi * phi
    for i in range(n_samples):
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        score = 0.0
        for j in range(n_features):
            score += x[j] * u[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 1.0)
        y_pred[i] = pred
        # Σ x
        Sigma_x = np.zeros(n_features)
        for j in range(n_features):
            for k in range(n_features):
                Sigma_x[j] += Sigma[j, k] * x[k]
        # v_t = x^T Σ x
        vt = 0.0
        for j in range(n_features):
            vt += x[j] * Sigma_x[j]
        mt = y_actual * score
        lt = phi * np.sqrt(vt) - mt
        if lt < 0.0:
            lt = 0.0
        if lt > 0.0:
            pa = 1.0 + phi_sq / 2.0
            xi = 1.0 + phi_sq
            sqrt_term = (mt * mt * phi_sq * phi_sq / 4.0) + (vt * phi_sq * xi)
            if sqrt_term < 0.0:
                sqrt_term = 0.0
            alpha_t = (1.0 / (vt * xi)) * (-mt * pa + np.sqrt(sqrt_term))
            if alpha_t < 0.0:
                alpha_t = 0.0
            if alpha_t > C:
                alpha_t = C
            sqrt_ut_term = (alpha_t * alpha_t * vt * vt * phi_sq) + (4.0 * vt)
            if sqrt_ut_term < 0.0:
                sqrt_ut_term = 0.0
            ut_val = 0.25 * (-alpha_t * vt * phi + np.sqrt(sqrt_ut_term)) ** 2
            beta_t = (alpha_t * phi) / (np.sqrt(ut_val) + vt * alpha_t * phi + 1e-8)
            for j in range(n_features):
                u[j] += alpha_t * y_actual * Sigma_x[j]
            for j in range(n_features):
                for k in range(n_features):
                    Sigma[j, k] -= beta_t * Sigma_x[j] * Sigma_x[k]
    return y_pred, u, Sigma


# ---------- 4i. AdaRDA (Adaptive RDA) ----------
@njit(cache=True)
def _adarda_kernel(X, y, perm, w, g, g1t, lambda_param, eta_param, delta_param, epoch, n_samples_total):
    """AdaRDA inner loop — RDA with per-feature adaptive step sizes.

    Idea: combines RDA's gradient-averaging sparsity with AdaGrad's
    trick of giving each feature its own learning rate.  Features
    that receive frequent large gradients get their step size
    automatically reduced (preventing overfit), while rare features
    keep a larger step size (allowing faster learning).  The result
    is a sparse model that also adapts to each feature's difficulty.

    Parameters
    ----------
    X               : ndarray (n, d) — training features
    y               : ndarray (n,)   — true labels in {-1, +1}
    perm            : ndarray (n,)   — index permutation
    w               : ndarray (d,)   — weight vector, reconstructed each step
    g               : ndarray (d,)   — running average gradient
    g1t             : ndarray (d,)   — cumulative squared gradients (AdaGrad)
    lambda_param    : float          — sparsity threshold
    eta_param       : float          — global step-size multiplier
    delta_param     : float          — AdaGrad denominator offset (stability)
    epoch           : int            — current epoch number
    n_samples_total : int            — samples per epoch

    Returns
    -------
    y_pred : ndarray (n,) — predictions before each update
    w      : ndarray (d,) — reconstructed weight vector
    g      : ndarray (d,) — updated gradient average
    g1t    : ndarray (d,) — updated squared-gradient accumulator
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.zeros(n_samples)
    for i in range(n_samples):
        t = i + 1 + epoch * n_samples_total
        idx = perm[i]
        x = X[idx]
        y_actual = y[idx]
        # Prediction
        score = 0.0
        for j in range(n_features):
            score += x[j] * w[j]
        pred = 1.0 if score > 0 else (-1.0 if score < 0 else 1.0)
        y_pred[i] = pred
        # Hinge loss
        margin = y_actual * score
        lt = 1.0 - margin
        if lt < 0.0:
            lt = 0.0
        # Gradient average and squared-gradient accumulator
        if lt > 0.0:
            for j in range(n_features):
                gt_j = -y_actual * x[j]
                g[j] = ((t - 1) / t) * g[j] + (1.0 / t) * gt_j
                g1t[j] += gt_j * gt_j
        else:
            for j in range(n_features):
                g[j] = ((t - 1) / t) * g[j]
        # Reconstruct weights with adaptive step
        for j in range(n_features):
            Ht_j = delta_param + np.sqrt(g1t[j])
            if np.abs(g[j]) > lambda_param:
                w[j] = np.sign(-g[j]) * eta_param * t / (Ht_j + 1e-8)
            else:
                w[j] = 0.0
    return y_pred, w, g, g1t


# ══════════════════════════════════════════════════════════════════════
#   SECTION 5 — ONLINE ALGORITHM WRAPPERS                              
#                                                                        
#   Each wrapper:                                                        
#     · Converts inputs to contiguous float64 for Numba                  
#     · Supports multi-epoch training with per-epoch shuffle             
#     · Optionally evaluates on a validation set each epoch              
#     · Returns (y_pred, best_weights)                                   
# ═══════════════════════════════════════════════════════════════════════

def PA(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Passive-Aggressive online classifier.

    Scans through the data one sample at a time.  When a sample is
    mis-classified (or classified with low confidence), the model
    makes the *minimum* weight change needed to fix it.  The name
    "passive-aggressive" comes from this: it is passive (no change)
    when correct, and aggressive (exact correction) when wrong.

    No hyperparameters to tune — fully automatic.

    Parameters
    ----------
    X, y       : training features and labels ({-1, +1})
    max_epochs : number of full passes over the data (1 = single pass)
    patience   : stop early if validation F1 hasn't improved for this many epochs
    X_val, y_val : optional hold-out set for early stopping

    Returns
    -------
    y_pred : ndarray — predictions on the training set (last epoch)
    w      : ndarray — best weight vector found
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    # Initialise weight vector to zeros
    w = np.zeros(n_features, dtype=np.float64)
    # Early-stopping bookkeeping
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    for epoch in range(max_epochs):
        # Multi-epoch: deterministic per-epoch shuffle (seed = 42 + epoch)
        if max_epochs > 1:
            np.random.seed(42 + epoch)
            perm = np.random.permutation(n_samples).astype(np.int64)
        else:
            # Single-pass: process samples in order
            perm = np.arange(n_samples, dtype=np.int64)
        # Run one full epoch through the Numba kernel
        y_pred, w = _pa_kernel(X_np, y_np, perm, w)
        final_weights = w.copy()
        # Validation evaluation (multi-epoch only)
        if max_epochs > 1 and X_val is not None and y_val is not None:
            y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), w)
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    # Return best weights found during validation, or final weights
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


def Perceptron(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Classic Perceptron online classifier.

    The oldest and simplest neural-network-style algorithm.  Whenever
    it makes a wrong prediction, it adds (or subtracts) the input
    features to (from) the weight vector.  Correct predictions cause
    no update at all.  No tuning parameters.

    Parameters
    ----------
    X, y       : training features and labels ({-1, +1})
    max_epochs : number of full passes over the data
    patience   : early-stopping patience (epochs)
    X_val, y_val : optional hold-out set

    Returns
    -------
    y_pred : ndarray — training predictions (last epoch)
    w      : ndarray — best weight vector found
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features, dtype=np.float64)
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    for epoch in range(max_epochs):
        if max_epochs > 1:
            np.random.seed(42 + epoch)
            perm = np.random.permutation(n_samples).astype(np.int64)
        else:
            perm = np.arange(n_samples, dtype=np.int64)
        y_pred, w = _percept_kernel(X_np, y_np, perm, w)
        final_weights = w.copy()
        if max_epochs > 1 and X_val is not None and y_val is not None:
            y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), w)
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


def OGC(X, y, max_epochs=1, patience=3, X_val=None, y_val=None):
    """Online Gradient Classification.

    Unlike Perceptron (updates only on mistakes), OGC adjusts weights
    on *every* sample using a normalised gradient step.  This makes it
    more responsive to the data distribution but also more sensitive
    to noise.  No tuning parameters.

    Parameters
    ----------
    X, y       : training features and labels ({-1, +1})
    max_epochs : number of full passes over the data
    patience   : early-stopping patience (epochs)
    X_val, y_val : optional hold-out set

    Returns
    -------
    y_pred : ndarray — training predictions (last epoch)
    w      : ndarray — best weight vector found
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features, dtype=np.float64)
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    for epoch in range(max_epochs):
        if max_epochs > 1:
            np.random.seed(42 + epoch)
            perm = np.random.permutation(n_samples).astype(np.int64)
        else:
            perm = np.arange(n_samples, dtype=np.int64)
        y_pred, w = _ogc_kernel(X_np, y_np, perm, w)
        final_weights = w.copy()
        if max_epochs > 1 and X_val is not None and y_val is not None:
            y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), w)
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


def AROW(X, y, r=0.1, max_epochs=1, patience=3, X_val=None, y_val=None,
         diagonal_sigma=False):
    """Adaptive Regularisation of Weights (AROW).

    Keeps a confidence score for each feature (or pair of features).
    Features the model is uncertain about receive larger updates;
    well-understood features get smaller updates.  This makes AROW
    robust to label noise — a single noisy sample won't ruin a
    feature the model has already learned well.

    Parameters
    ----------
    X, y           : training features and labels ({-1, +1})
    r              : regularisation strength (larger = more cautious updates)
    max_epochs     : number of full passes over the data
    patience       : early-stopping patience (epochs)
    X_val, y_val   : optional hold-out set
    diagonal_sigma : True → O(d) per-feature uncertainty (fast, good default)
                     False → O(d²) full covariance (captures feature correlations)

    Returns
    -------
    y_pred : ndarray — training predictions (last epoch)
    w      : ndarray — best weight vector found
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    u = np.zeros(n_features, dtype=np.float64)
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    if diagonal_sigma:
        # Per-feature variance vector (starts at 1 = maximum uncertainty)
        sigma_diag = np.ones(n_features, dtype=np.float64)
        for epoch in range(max_epochs):
            if max_epochs > 1:
                np.random.seed(42 + epoch)
                perm = np.random.permutation(n_samples).astype(np.int64)
            else:
                perm = np.arange(n_samples, dtype=np.int64)
            y_pred, u, sigma_diag = _arow_diag_kernel(X_np, y_np, perm, u, sigma_diag, r)
            final_weights = u.copy()
            if max_epochs > 1 and X_val is not None and y_val is not None:
                y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), u)
                _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = u.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
                    break
    else:
        # Full d × d covariance (identity = max uncertainty)
        Sigma = np.eye(n_features, dtype=np.float64)
        for epoch in range(max_epochs):
            if max_epochs > 1:
                np.random.seed(42 + epoch)
                perm = np.random.permutation(n_samples).astype(np.int64)
            else:
                perm = np.arange(n_samples, dtype=np.int64)
            y_pred, u, Sigma = _arow_full_kernel(X_np, y_np, perm, u, Sigma, r)
            final_weights = u.copy()
            if max_epochs > 1 and X_val is not None and y_val is not None:
                y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), u)
                _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = u.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1} (patience={patience})")
                    break
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


def RDA(X, y, lambda_param=0.0001, gamma_param=1, max_epochs=1, patience=3,
        X_val=None, y_val=None):
    """Regularised Dual Averaging — learns sparse (few-feature) models.

    Instead of updating weights directly, RDA accumulates gradient
    information over time and reconstructs weights by thresholding.
    Features with consistently small gradients are forced to exactly
    zero — effectively removing them from the model.  Good when you
    suspect many features are irrelevant.

    Parameters
    ----------
    X, y         : training features and labels ({-1, +1})
    lambda_param : sparsity threshold — higher means more features set to zero
                   (0.0001 works for standardised data; 1.0 zeros everything)
    gamma_param  : step-size denominator (controls learning speed)
    max_epochs   : number of full passes over the data
    patience     : early-stopping patience (epochs)
    X_val, y_val : optional hold-out set

    Returns
    -------
    y_pred : ndarray — training predictions (last epoch)
    w      : ndarray — best weight vector found (sparse)
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features, dtype=np.float64)
    # Running average gradient
    g = np.zeros(n_features, dtype=np.float64)
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    for epoch in range(max_epochs):
        if max_epochs > 1:
            np.random.seed(42 + epoch)
            perm = np.random.permutation(n_samples).astype(np.int64)
        else:
            perm = np.arange(n_samples, dtype=np.int64)
        y_pred, w, g = _rda_kernel(X_np, y_np, perm, w, g, lambda_param, gamma_param, epoch, n_samples)
        final_weights = w.copy()
        if max_epochs > 1 and X_val is not None and y_val is not None:
            y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), w)
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


def SCW(X, y, C=1, eta=0.9, max_epochs=1, patience=3, X_val=None, y_val=None,
        diagonal_sigma=False):
    """Soft Confidence-Weighted learning (SCW-I).

    Like AROW, SCW tracks per-feature uncertainty.  The key twist is
    that SCW triggers updates even on *correct* predictions if the
    model is not confident enough.  The confidence threshold is set
    by eta (e.g. 0.9 = "90% confident").  C caps the maximum step
    size to prevent over-correction on outliers.

    Parameters
    ----------
    X, y           : training features and labels ({-1, +1})
    C              : aggressiveness cap (max step size per update)
    eta            : confidence level in (0.5, 1.0)
                     0.9 = require 90% confidence 
                     0.5 = no confidence requirement 
    max_epochs     : number of full passes over the data
    patience       : early-stopping patience (epochs)
    X_val, y_val   : optional hold-out set
    diagonal_sigma : True for high-dimensional data (faster)

    Returns
    -------
    y_pred : ndarray — training predictions (last epoch)
    w      : ndarray — best weight vector found
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    # Convert confidence level η to probit φ
    phi = float(norm.ppf(eta))
    u = np.zeros(n_features, dtype=np.float64)
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    if diagonal_sigma:
        sigma_diag = np.ones(n_features, dtype=np.float64)
        for epoch in range(max_epochs):
            if max_epochs > 1:
                np.random.seed(42 + epoch)
                perm = np.random.permutation(n_samples).astype(np.int64)
            else:
                perm = np.arange(n_samples, dtype=np.int64)
            y_pred, u, sigma_diag = _scw_diag_kernel(X_np, y_np, perm, u, sigma_diag, C, phi)
            final_weights = u.copy()
            if max_epochs > 1 and X_val is not None and y_val is not None:
                y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), u)
                _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = u.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
    else:
        Sigma = np.eye(n_features, dtype=np.float64)
        for epoch in range(max_epochs):
            if max_epochs > 1:
                np.random.seed(42 + epoch)
                perm = np.random.permutation(n_samples).astype(np.int64)
            else:
                perm = np.arange(n_samples, dtype=np.int64)
            y_pred, u, Sigma = _scw_full_kernel(X_np, y_np, perm, u, Sigma, C, phi)
            final_weights = u.copy()
            if max_epochs > 1 and X_val is not None and y_val is not None:
                y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), u)
                _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_weights = u.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


def AdaRDA(X, y, lambda_param=1, eta_param=1, delta_param=1,
           max_epochs=1, patience=3, X_val=None, y_val=None):
    """Adaptive RDA — sparse learning with per-feature learning rates.

    Combines RDA's ability to zero out irrelevant features with
    AdaGrad's trick of giving each feature its own learning rate.
    Features that appear frequently get a slower rate (preventing
    overfit); rare features keep a fast rate (allowing quicker learning).
    Good for datasets with many features of varying importance.

    Parameters
    ----------
    X, y         : training features and labels ({-1, +1})
    lambda_param : sparsity threshold (higher = more features zeroed)
    eta_param    : global step-size multiplier
    delta_param  : AdaGrad offset — prevents division by zero in step-size calc
    max_epochs   : number of full passes over the data
    patience     : early-stopping patience (epochs)
    X_val, y_val : optional hold-out set

    Returns
    -------
    y_pred : ndarray — training predictions (last epoch)
    w      : ndarray — best weight vector found (sparse)
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    n_samples, n_features = X_np.shape
    w = np.zeros(n_features, dtype=np.float64)
    g = np.zeros(n_features, dtype=np.float64)
    # AdaGrad squared-gradient accumulator
    g1t = np.zeros(n_features, dtype=np.float64)
    best_val_f1 = -1.0
    patience_counter = 0
    best_weights = None
    final_weights = None

    for epoch in range(max_epochs):
        if max_epochs > 1:
            np.random.seed(42 + epoch)
            perm = np.random.permutation(n_samples).astype(np.int64)
        else:
            perm = np.arange(n_samples, dtype=np.int64)
        y_pred, w, g, g1t = _adarda_kernel(
            X_np, y_np, perm, w, g, g1t,
            lambda_param, eta_param, delta_param, epoch, n_samples,
        )
        final_weights = w.copy()
        if max_epochs > 1 and X_val is not None and y_val is not None:
            y_pred_val = _predict_val(np.ascontiguousarray(X_val, dtype=np.float64), w)
            _, _, _, _, val_f1 = calculate_class1_metrics(y_val, y_pred_val)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = w.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            print(f"    Epoch {epoch+1}: Val F1 = {val_f1:.4f} (best: {best_val_f1:.4f})")
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    w_out = best_weights if best_weights is not None else final_weights
    return y_pred, w_out


# ═══════════════════════════════════════════════════════════════════════
#   JIT WARM-UP                                                         
# ═══════════════════════════════════════════════════════════════════════

def warmup_jit():
    """Pre-compile every Numba kernel with tiny dummy data.

    Call once before timing experiments so that Numba compilation
    overhead is not included in measured run times.
    """
    print("Warming up Numba JIT kernels...")
    n, d = 4, 3
    X = np.random.randn(n, d)
    y = np.array([1.0, -1.0, 1.0, -1.0])
    perm = np.arange(n, dtype=np.int64)
    # Compile each kernel with minimal data
    _pa_kernel(X, y, perm, np.zeros(d))
    _percept_kernel(X, y, perm, np.zeros(d))
    _ogc_kernel(X, y, perm, np.zeros(d))
    _rda_kernel(X, y, perm, np.zeros(d), np.zeros(d), 1.0, 1.0, 0, n)
    _adarda_kernel(X, y, perm, np.zeros(d), np.zeros(d), np.zeros(d), 1.0, 1.0, 1.0, 0, n)
    _arow_diag_kernel(X, y, perm, np.zeros(d), np.ones(d), 0.1)
    _arow_full_kernel(X, y, perm, np.zeros(d), np.eye(d), 0.1)
    _scw_diag_kernel(X, y, perm, np.zeros(d), np.ones(d), 1.0, 0.0)
    _scw_full_kernel(X, y, perm, np.zeros(d), np.eye(d), 1.0, 0.0)
    _predict_val(X, np.zeros(d))
    print("JIT warm-up complete.")


# ═══════════════════════════════════════════════════════════════════════
#   NUMBA HELPERS — shared by Sections 6 and 7                          
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _rbf_vec(x_i, X_sv, gamma, n_sv, d):
    """Compute k(x_i, X_sv[j]) for j in 0..n_sv-1. Returns (n_sv,) array."""
    k = np.empty(n_sv)
    for j in range(n_sv):
        s = 0.0
        for f in range(d):
            diff = x_i[f] - X_sv[j, f]
            s += diff * diff
        k[j] = np.exp(-gamma * s)
    return k


@njit(cache=True)
def _rbf_self(x_i, gamma, d):
    """k(x_i, x_i) — always 1.0 for RBF, but kept for generality."""
    return 1.0


@njit(cache=True)
def _find_idx(sv_ids, n_sv, val):
    """Linear scan to find val in sv_ids[0:n_sv]. Returns index or -1."""
    for j in range(n_sv):
        if sv_ids[j] == val:
            return j
    return -1


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 6 — KERNEL ONLINE ALGORITHMS  (Numba-accelerated)          
#                                                                        
#   Single-pass over data.  Each sample that triggers an update is       
#   stored as a support vector.  Prediction at sample i uses all SVs     
#   collected so far: f(x) = Σ_j α_j k(x, sv_j).                       
#   Returns y_pred array only (no weight vector in kernel space).        
# ═══════════════════════════════════════════════════════════════════════

# --------------- Numba inner kernels (kernel-online) ---------------

@njit(cache=True)
def _kpa_online(X, y, gamma):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    n_sv = 0
    y_pred = np.empty(n)
    for i in range(n):
        if n_sv == 0:
            f_t = 1.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
        pred = 1.0 if f_t >= 0 else -1.0
        y_pred[i] = pred
        loss = 1.0 - y[i] * pred
        if loss < 0.0:
            loss = 0.0
        if loss > 0.0:
            tau = loss / (1.0 + 1e-8)
            sv_alpha[n_sv] = tau * y[i]
            for f in range(d):
                sv_X[n_sv, f] = X[i, f]
            n_sv += 1
    return y_pred


@njit(cache=True)
def _kperceptron_online(X, y, gamma):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    n_sv = 0
    y_pred = np.empty(n)
    for i in range(n):
        if n_sv == 0:
            f_t = 1.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
        pred = 1.0 if f_t >= 0 else -1.0
        y_pred[i] = pred
        if pred != y[i]:
            sv_alpha[n_sv] = y[i]
            for f in range(d):
                sv_X[n_sv, f] = X[i, f]
            n_sv += 1
    return y_pred


@njit(cache=True)
def _kogc_online(X, y, gamma):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    n_sv = 0
    y_pred = np.empty(n)
    for i in range(n):
        if n_sv == 0:
            f_t = 0.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
        if f_t > 0:
            raw_pred = 1.0
        elif f_t < 0:
            raw_pred = -1.0
        else:
            raw_pred = 0.0
        y_pred[i] = raw_pred if raw_pred != 0 else 1.0
        error = y[i] - raw_pred
        eta = error / (1.0 + 1e-8)
        sv_alpha[n_sv] = eta
        for f in range(d):
            sv_X[n_sv, f] = X[i, f]
        n_sv += 1
    return y_pred


@njit(cache=True)
def _karow_online(X, y, gamma, r):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    sv_sigma = np.empty(n)
    n_sv = 0
    y_pred = np.empty(n)
    for i in range(n):
        if n_sv == 0:
            f_t = 1.0
            v_t = 1.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
            v_t = 0.0
            for j in range(n_sv):
                v_t += k[j] * k[j] * sv_sigma[j]
            v_t += 1e-8
        pred = 1.0 if f_t >= 0 else -1.0
        y_pred[i] = pred
        lt = 1.0 - y[i] * f_t
        if lt < 0.0:
            lt = 0.0
        if lt > 0.0:
            alpha_t = lt / (v_t + r)
            beta_t = 1.0 / (v_t + r)
            sv_alpha[n_sv] = alpha_t * y[i]
            sv_sigma[n_sv] = 1.0 / (1.0 + beta_t)
            for f in range(d):
                sv_X[n_sv, f] = X[i, f]
            n_sv += 1
    return y_pred


@njit(cache=True)
def _krda_online(X, y, gamma, lambda_param, gamma_param):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    n_sv = 0
    g = 0.0
    y_pred = np.empty(n)
    for i in range(n):
        t = float(i + 1)
        if n_sv == 0:
            f_t = 1.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
        pred = 1.0 if f_t >= 0 else -1.0
        y_pred[i] = pred
        lt = 1.0 - y[i] * f_t
        if lt < 0.0:
            lt = 0.0
        gt = -y[i] if lt > 0.0 else 0.0
        g = ((t - 1.0) / t) * g + (1.0 / t) * gt
        if abs(g) > lambda_param:
            sign_g = 1.0 if g > 0 else -1.0
            coeff = -(np.sqrt(t) / gamma_param) * (g - lambda_param * sign_g)
        else:
            coeff = 0.0
        if coeff != 0.0:
            sv_alpha[n_sv] = coeff * y[i]
            for f in range(d):
                sv_X[n_sv, f] = X[i, f]
            n_sv += 1
    return y_pred


@njit(cache=True)
def _kscw_online(X, y, gamma, C, phi):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    sv_sigma = np.empty(n)
    n_sv = 0
    y_pred = np.empty(n)
    for i in range(n):
        if n_sv == 0:
            f_t = 1.0
            v_t = 1.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
            v_t = 0.0
            for j in range(n_sv):
                v_t += k[j] * k[j] * sv_sigma[j]
            v_t += 1e-8
        pred = 1.0 if f_t >= 0 else -1.0
        y_pred[i] = pred
        m_t = y[i] * f_t
        lt = phi * np.sqrt(v_t) - m_t
        if lt < 0.0:
            lt = 0.0
        if lt > 0.0:
            pa = 1.0 + (phi * phi) / 2.0
            xi = 1.0 + phi * phi
            sqrt_term = (m_t * m_t * phi ** 4 / 4.0) + (v_t * phi * phi * xi)
            if sqrt_term < 0.0:
                sqrt_term = 0.0
            a_t = (1.0 / (v_t * xi + 1e-8)) * (-m_t * pa + np.sqrt(sqrt_term))
            if a_t < 0.0:
                a_t = 0.0
            if a_t > C:
                a_t = C
            sqrt_ut = (a_t * a_t * v_t * v_t * phi * phi) + (4.0 * v_t)
            if sqrt_ut < 0.0:
                sqrt_ut = 0.0
            ut = 0.25 * (-a_t * v_t * phi + np.sqrt(sqrt_ut)) ** 2
            beta_t = (a_t * phi) / (np.sqrt(ut) + v_t * a_t * phi + 1e-8)
            sv_alpha[n_sv] = a_t * y[i]
            sv_sigma[n_sv] = 1.0 / (1.0 + beta_t)
            for f in range(d):
                sv_X[n_sv, f] = X[i, f]
            n_sv += 1
    return y_pred


@njit(cache=True)
def _kadarda_online(X, y, gamma, lambda_param, eta_param, delta_param):
    n, d = X.shape
    sv_X = np.empty((n, d))
    sv_alpha = np.empty(n)
    n_sv = 0
    g = 0.0
    g1t = 0.0
    y_pred = np.empty(n)
    for i in range(n):
        t = float(i + 1)
        if n_sv == 0:
            f_t = 1.0
        else:
            k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
            f_t = 0.0
            for j in range(n_sv):
                f_t += k[j] * sv_alpha[j]
        pred = 1.0 if f_t >= 0 else -1.0
        y_pred[i] = pred
        lt = 1.0 - y[i] * f_t
        if lt < 0.0:
            lt = 0.0
        gt = -y[i] if lt > 0.0 else 0.0
        g = ((t - 1.0) / t) * g + (1.0 / t) * gt
        g1t += gt * gt
        Ht = delta_param + np.sqrt(g1t)
        if abs(g) > lambda_param:
            sign_g = 1.0 if g > 0 else (-1.0 if g < 0 else 0.0)
            coeff = sign_g * (-1.0) * eta_param * t / (Ht + 1e-8)
        else:
            coeff = 0.0
        if coeff != 0.0:
            sv_alpha[n_sv] = coeff * y[i]
            for f in range(d):
                sv_X[n_sv, f] = X[i, f]
            n_sv += 1
    return y_pred


# --------------- Wrapper functions ---------------

def KPA(X, y, gamma=1.0):
    """Kernel Passive-Aggressive — single-pass PA in kernel (similarity) space.

    Works like PA but instead of raw features, decisions are based on
    how similar the current sample is to previously stored "support
    vectors".  This lets it learn non-linear (curved) boundaries.
    Every mis-classified sample is saved as a new support vector.
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _kpa_online(X_np, y_np, gamma)


def KPerceptron(X, y, gamma=1.0):
    """Kernel Perceptron — single-pass Perceptron in similarity space.

    Adds the sample as a support vector whenever the prediction is wrong.
    Uses RBF similarity to make decisions, enabling non-linear boundaries.
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _kperceptron_online(X_np, y_np, gamma)


def KOGC(X, y, gamma=1.0):
    """Kernel OGC — gradient step on every sample in similarity space.

    Every sample becomes a support vector.  The coefficient for each
    is proportional to the error normalised by the self-similarity.
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _kogc_online(X_np, y_np, gamma)


def KAROW(X, y, r=0.1, gamma=1.0):
    """Kernel AROW — confidence-aware learning in similarity space.

    Combines AROW's per-component uncertainty tracking with kernel
    similarity.  Each support vector gets its own uncertainty score.
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _karow_online(X_np, y_np, gamma, r)


def KRDA(X, y, lambda_param=0.01, gamma_param=1, gamma=1.0):
    """Kernel RDA — L1-regularised dual averaging in kernel space.

    Scalar running-average gradient. New SV added when |g| > lambda.
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _krda_online(X_np, y_np, gamma, lambda_param, float(gamma_param))


def KSCW(X, y, C=1, eta=0.9, gamma=1.0):
    """Kernel SCW — confidence-weighted learning in similarity space.

    Like KAROW but with a confidence-based loss: updates happen even
    on correct predictions if the model is unsure.
    """
    from scipy.stats import norm
    phi = norm.ppf(eta)
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _kscw_online(X_np, y_np, gamma, float(C), phi)


def KAdaRDA(X, y, lambda_param=0.01, eta_param=1, delta_param=1, gamma=1.0):
    """Kernel AdaRDA — adaptive sparse learning in similarity space.

    Combines KRDA's sparsity with per-sample adaptive step sizes.
    """
    X_np = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_np = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    return _kadarda_online(X_np, y_np, gamma, lambda_param, float(eta_param), float(delta_param))


# ═══════════════════════════════════════════════════════════════════════
#   SECTION 7 — BATCH KERNEL ALGORITHMS  (Numba-accelerated)           
#                                                                        
#   Class-based.  Each class has fit(X, y, epochs), predict(X), and      
#   decision_function(X).  Inner loops compiled via @njit for speed.     
#   RBF kernel computed inline — no Python-level per-sample calls.       
# ═══════════════════════════════════════════════════════════════════════


# --------------- Numba inner loops ---------------

@njit(cache=True)
def _batch_perceptron_fit(X, y, gamma, max_sv, epochs):
    n, d = X.shape
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    n_sv = 0

    for epoch in range(epochs):
        for i in range(n):
            # predict
            pred = 0.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    pred += k[j] * sv_alpha[j]
            pred_label = 1.0 if pred >= 0 else -1.0
            if pred_label != y[i]:
                # add or update SV
                if n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_alpha[n_sv] = y[i]
                    n_sv += 1
                else:
                    # shift left (drop oldest)
                    for j in range(n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                    for f in range(d):
                        sv_X[n_sv - 1, f] = X[i, f]
                    sv_alpha[n_sv - 1] = y[i]
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


@njit(cache=True)
def _batch_pa_fit(X, y, gamma, max_sv, C, epochs):
    n, d = X.shape
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    sv_ids = np.full(max_sv, -1, dtype=np.int64)
    n_sv = 0

    for epoch in range(epochs):
        for i in range(n):
            pred = 0.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    pred += k[j] * sv_alpha[j]
            loss = 1.0 - y[i] * pred
            if loss < 0.0:
                loss = 0.0
            if loss > 0.0:
                tau = loss / (1.0 + 1e-8)
                if tau > C:
                    tau = C
                idx = _find_idx(sv_ids, n_sv, i)
                if idx >= 0:
                    sv_alpha[idx] += tau * y[i]
                elif n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_alpha[n_sv] = tau * y[i]
                    sv_ids[n_sv] = i
                    n_sv += 1
                else:
                    # drop oldest
                    for j in range(n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                        sv_ids[j] = sv_ids[j + 1]
                    for f in range(d):
                        sv_X[n_sv - 1, f] = X[i, f]
                    sv_alpha[n_sv - 1] = tau * y[i]
                    sv_ids[n_sv - 1] = i
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


@njit(cache=True)
def _batch_arow_fit(X, y, gamma, max_sv, r, epochs):
    n, d = X.shape
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    sv_sigma = np.ones(max_sv)  # diagonal covariance
    sv_ids = np.full(max_sv, -1, dtype=np.int64)
    n_sv = 0

    for epoch in range(epochs):
        for i in range(n):
            pred = 0.0
            v_t = 1.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    pred += k[j] * sv_alpha[j]
                v_t = 0.0
                for j in range(n_sv):
                    v_t += k[j] * sv_sigma[j] * k[j]
            loss = 1.0 - y[i] * pred
            if loss < 0.0:
                loss = 0.0
            if loss > 0.0:
                beta_t = 1.0 / (v_t + r)
                alpha_t = loss * beta_t
                idx = _find_idx(sv_ids, n_sv, i)
                if idx >= 0:
                    sv_alpha[idx] += alpha_t * y[i]
                    k2 = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                    for j in range(n_sv):
                        sv_sigma[j] -= beta_t * (sv_sigma[j] * k2[j]) ** 2
                        if sv_sigma[j] < 1e-10:
                            sv_sigma[j] = 1e-10
                elif n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_alpha[n_sv] = alpha_t * y[i]
                    sv_sigma[n_sv] = 1.0
                    sv_ids[n_sv] = i
                    n_sv += 1
                else:
                    for j in range(n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                        sv_sigma[j] = sv_sigma[j + 1]
                        sv_ids[j] = sv_ids[j + 1]
                    for f in range(d):
                        sv_X[n_sv - 1, f] = X[i, f]
                    sv_alpha[n_sv - 1] = alpha_t * y[i]
                    sv_sigma[n_sv - 1] = 1.0
                    sv_ids[n_sv - 1] = i
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


@njit(cache=True)
def _batch_gc_fit(X, y, gamma, max_sv, pos_weight, neg_weight, epochs):
    n, d = X.shape
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    sv_ids = np.full(max_sv, -1, dtype=np.int64)
    sv_grad_sum = np.zeros(max_sv)
    n_sv = 0
    t = 0

    for epoch in range(epochs):
        for i in range(n):
            t += 1
            score = 0.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    score += k[j] * sv_alpha[j]
            margin = y[i] * score
            if margin < 1.0:
                w = pos_weight if y[i] == 1.0 else neg_weight
                grad = -y[i] * w
            else:
                grad = 0.0

            idx = _find_idx(sv_ids, n_sv, i)
            if idx >= 0:
                sv_grad_sum[idx] += grad
                avg_grad = sv_grad_sum[idx] / t
                sv_alpha[idx] = -np.sqrt(t) * avg_grad
            elif grad != 0.0:
                if n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_grad_sum[n_sv] = grad
                    avg_grad = grad / t
                    sv_alpha[n_sv] = -np.sqrt(t) * avg_grad
                    sv_ids[n_sv] = i
                    n_sv += 1
                # budget: remove smallest |alpha|
                if n_sv > max_sv:
                    min_j = 0
                    min_val = abs(sv_alpha[0])
                    for j in range(1, n_sv):
                        if abs(sv_alpha[j]) < min_val:
                            min_val = abs(sv_alpha[j])
                            min_j = j
                    # shift left to remove min_j
                    for j in range(min_j, n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                        sv_ids[j] = sv_ids[j + 1]
                        sv_grad_sum[j] = sv_grad_sum[j + 1]
                    n_sv -= 1
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


@njit(cache=True)
def _batch_rda_fit(X, y, gamma, max_sv, lambda_param, rda_gamma, epochs):
    n, d = X.shape
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    sv_ids = np.full(max_sv, -1, dtype=np.int64)
    sv_grad_sum = np.zeros(max_sv)
    n_sv = 0
    t = 0

    for epoch in range(epochs):
        for i in range(n):
            t += 1
            pred = 0.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    pred += k[j] * sv_alpha[j]
            loss = 1.0 - y[i] * pred
            if loss < 0.0:
                loss = 0.0
            grad = -y[i] if loss > 0.0 else 0.0

            idx = _find_idx(sv_ids, n_sv, i)
            if idx >= 0:
                sv_grad_sum[idx] += grad
                avg_grad = sv_grad_sum[idx] / t
                if abs(avg_grad) > lambda_param:
                    sign_g = 1.0 if avg_grad > 0 else -1.0
                    sv_alpha[idx] = -(np.sqrt(t) / rda_gamma) * (avg_grad - lambda_param * sign_g)
                else:
                    sv_alpha[idx] = 0.0
            elif grad != 0.0:
                if n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_grad_sum[n_sv] = grad
                    avg_grad = grad / t
                    if abs(avg_grad) > lambda_param:
                        sign_g = 1.0 if avg_grad > 0 else -1.0
                        sv_alpha[n_sv] = -(np.sqrt(t) / rda_gamma) * (avg_grad - lambda_param * sign_g)
                    else:
                        sv_alpha[n_sv] = 0.0
                    sv_ids[n_sv] = i
                    n_sv += 1
                if n_sv > max_sv:
                    # drop oldest
                    for j in range(n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                        sv_ids[j] = sv_ids[j + 1]
                        sv_grad_sum[j] = sv_grad_sum[j + 1]
                    n_sv -= 1
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


@njit(cache=True)
def _batch_scw_fit(X, y, gamma, max_sv, C, phi, epochs):
    n, d = X.shape
    psi = 1.0 + phi * phi / 2.0
    zeta = 1.0 + phi * phi
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    sv_sigma = np.ones(max_sv)  # diagonal covariance
    sv_ids = np.full(max_sv, -1, dtype=np.int64)
    n_sv = 0

    for epoch in range(epochs):
        for i in range(n):
            pred = 0.0
            v_t = 1.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    pred += k[j] * sv_alpha[j]
                v_t = 0.0
                for j in range(n_sv):
                    v_t += k[j] * sv_sigma[j] * k[j]
            m_t = y[i] * pred
            loss = phi * np.sqrt(v_t) - m_t
            if loss > 0.0:
                disc = m_t * m_t * phi ** 4 / 4.0 + v_t * phi * phi * zeta
                if disc < 0.0:
                    disc = 0.0
                alpha_t = (-m_t * psi + np.sqrt(disc)) / (v_t * zeta + 1e-8)
                if alpha_t < 0.0:
                    alpha_t = 0.0
                if alpha_t > C:
                    alpha_t = C
                u_disc = alpha_t * alpha_t * v_t * v_t * phi * phi + 4.0 * v_t
                if u_disc < 0.0:
                    u_disc = 0.0
                u_t = 0.25 * (-alpha_t * v_t * phi + np.sqrt(u_disc)) ** 2
                beta_t = alpha_t * phi / (np.sqrt(u_t) + v_t * alpha_t * phi + 1e-8)

                idx = _find_idx(sv_ids, n_sv, i)
                if idx >= 0:
                    sv_alpha[idx] += alpha_t * y[i]
                    k2 = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                    for j in range(n_sv):
                        sv_sigma[j] -= beta_t * (sv_sigma[j] * k2[j]) ** 2
                        if sv_sigma[j] < 1e-10:
                            sv_sigma[j] = 1e-10
                elif n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_alpha[n_sv] = alpha_t * y[i]
                    sv_sigma[n_sv] = 1.0
                    sv_ids[n_sv] = i
                    n_sv += 1
                else:
                    for j in range(n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                        sv_sigma[j] = sv_sigma[j + 1]
                        sv_ids[j] = sv_ids[j + 1]
                    for f in range(d):
                        sv_X[n_sv - 1, f] = X[i, f]
                    sv_alpha[n_sv - 1] = alpha_t * y[i]
                    sv_sigma[n_sv - 1] = 1.0
                    sv_ids[n_sv - 1] = i
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


@njit(cache=True)
def _batch_adardr_fit(X, y, gamma, max_sv, lambda_param, delta, epochs):
    n, d = X.shape
    sv_X = np.empty((max_sv, d))
    sv_alpha = np.empty(max_sv)
    sv_ids = np.full(max_sv, -1, dtype=np.int64)
    sv_grad_sum = np.zeros(max_sv)
    sv_sq_grad_sum = np.zeros(max_sv)
    n_sv = 0

    for epoch in range(epochs):
        for i in range(n):
            pred = 0.0
            if n_sv > 0:
                k = _rbf_vec(X[i], sv_X, gamma, n_sv, d)
                for j in range(n_sv):
                    pred += k[j] * sv_alpha[j]
            loss = 1.0 - y[i] * pred
            if loss < 0.0:
                loss = 0.0
            grad = -y[i] if loss > 0.0 else 0.0

            idx = _find_idx(sv_ids, n_sv, i)
            if idx >= 0:
                sv_grad_sum[idx] += grad
                sv_sq_grad_sum[idx] += grad * grad
                H_t = delta + np.sqrt(sv_sq_grad_sum[idx])
                avg_g = sv_grad_sum[idx]
                if abs(avg_g) > lambda_param * H_t:
                    sign_g = 1.0 if avg_g > 0 else -1.0
                    sv_alpha[idx] = -(1.0 / H_t) * (avg_g - lambda_param * H_t * sign_g)
                else:
                    sv_alpha[idx] = 0.0
            elif grad != 0.0:
                if n_sv < max_sv:
                    for f in range(d):
                        sv_X[n_sv, f] = X[i, f]
                    sv_grad_sum[n_sv] = grad
                    sv_sq_grad_sum[n_sv] = grad * grad
                    H_t = delta + np.sqrt(sv_sq_grad_sum[n_sv])
                    avg_g = sv_grad_sum[n_sv]
                    if abs(avg_g) > lambda_param * H_t:
                        sign_g = 1.0 if avg_g > 0 else -1.0
                        sv_alpha[n_sv] = -(1.0 / H_t) * (avg_g - lambda_param * H_t * sign_g)
                    else:
                        sv_alpha[n_sv] = 0.0
                    sv_ids[n_sv] = i
                    n_sv += 1
                if n_sv > max_sv:
                    for j in range(n_sv - 1):
                        for f in range(d):
                            sv_X[j, f] = sv_X[j + 1, f]
                        sv_alpha[j] = sv_alpha[j + 1]
                        sv_ids[j] = sv_ids[j + 1]
                        sv_grad_sum[j] = sv_grad_sum[j + 1]
                        sv_sq_grad_sum[j] = sv_sq_grad_sum[j + 1]
                    n_sv -= 1
    return sv_X[:n_sv].copy(), sv_alpha[:n_sv].copy()


# --------------- Class wrappers ---------------

class KernelPerceptron:
    """Batch Kernel Perceptron — Numba-accelerated single/multi-pass."""

    def __init__(self, gamma=0.1, max_sv=1000):
        self.gamma = gamma
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        sv, al = _batch_perceptron_fit(X_c, y_c, self.gamma, self.max_sv, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class KernelPA:
    """Batch Kernel PA-I — Numba-accelerated."""

    def __init__(self, gamma=0.1, max_sv=1000, C=1.0):
        self.gamma = gamma
        self.max_sv = max_sv
        self.C = C
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        sv, al = _batch_pa_fit(X_c, y_c, self.gamma, self.max_sv, self.C, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class KernelAROW:
    """Batch Kernel AROW — Numba-accelerated with diagonal covariance."""

    def __init__(self, gamma=0.1, r=1.0, max_sv=500):
        self.gamma = gamma
        self.r = r
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        sv, al = _batch_arow_fit(X_c, y_c, self.gamma, self.max_sv, self.r, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class KernelGC:
    """Batch Kernel Gradient Classification — Numba-accelerated."""

    def __init__(self, gamma=0.1, max_sv=1000):
        self.gamma = gamma
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        n_pos = np.sum(y_c == 1.0)
        n_neg = np.sum(y_c == -1.0)
        n = len(y_c)
        pw = n / (2.0 * n_pos) if n_pos > 0 else 1.0
        nw = n / (2.0 * n_neg) if n_neg > 0 else 1.0
        sv, al = _batch_gc_fit(X_c, y_c, self.gamma, self.max_sv, pw, nw, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class KernelRDA:
    """Batch Kernel RDA — Numba-accelerated sparse model."""

    def __init__(self, gamma=0.1, lambda_param=0.01, rda_gamma=1.0, max_sv=500):
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.rda_gamma = rda_gamma
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        sv, al = _batch_rda_fit(X_c, y_c, self.gamma, self.max_sv,
                                self.lambda_param, self.rda_gamma, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class KernelSCW:
    """Batch Kernel SCW-I — Numba-accelerated with diagonal covariance."""

    def __init__(self, gamma=0.1, C=1.0, phi=0.5, max_sv=500):
        self.gamma = gamma
        self.C = C
        self.phi = phi
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        sv, al = _batch_scw_fit(X_c, y_c, self.gamma, self.max_sv,
                                self.C, self.phi, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class KernelAdaRDA:
    """Batch Kernel AdaRDA — Numba-accelerated adaptive sparse model."""

    def __init__(self, gamma=0.1, lambda_param=0.01, delta=1.0, max_sv=500):
        self.gamma = gamma
        self.lambda_param = lambda_param
        self.delta = delta
        self.max_sv = max_sv
        self.support_vectors = None
        self.alpha = None

    def fit(self, X, y, epochs=3):
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        sv, al = _batch_adardr_fit(X_c, y_c, self.gamma, self.max_sv,
                                   self.lambda_param, self.delta, epochs)
        self.support_vectors = sv
        self.alpha = al
        return self

    def decision_function(self, X):
        if self.support_vectors is None or len(self.support_vectors) == 0:
            return np.zeros(len(X))
        return rbf_kernel_matrix(X, self.support_vectors, self.gamma) @ self.alpha

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
