"""
Neural-network replacement for the Volterra kernel, starting with 2 outputs
(Cl, Cm) and designed to extend to 2+r outputs later (just add names to
`output_names` and rebuild the dataset targets).

Architecture: causal, dilated 1D-convolutional backbone (TCN, Bai et al.
2018 style) -- the natural nonlinear generalization of a fixed-memory-depth
Volterra kernel -- with a shared trunk and one small head per output.
Includes:
  - lagged-window dataset construction from pitch_input / zdot_input
  - variance-normalized multi-output loss (avoids the same amplitude-scaling
    problem we hit with the joint SLSQP Volterra fit)
  - optional residual-on-top-of-linear-Volterra hybrid mode
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


import csv 

def read_csv(filename):
    with open(filename, newline='') as f_input:
        next(f_input)
        next(f_input)
        return [list(map(float, row)) for row in csv.reader(f_input)]


def generate_schroeder_multisine(freqs, dt, duration, amp=1.0):
    t = np.arange(0, duration, dt)
    N = len(freqs)
    # Schroeder Phase Formula: phi_k = -k*(k-1)*pi / N
    phases = [-k * (k - 1) * np.pi / N for k in range(  1, N + 1)]
    
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += amp * np.sin(2 * np.pi * f * t + phases[i])
    
    # Normalize to keep within original amplitude constraints
    # signal = signal / np.max(np.abs(signal)) * amp
    return t, signal


# ============================================================================
# 1) Build lagged input windows from pitch_input / zdot_input
# ============================================================================
 
def build_lagged_windows(inputs, m):
    """
    Parameters
    ----------
    inputs : list of 1D ndarrays, each shape (n_time,)
        e.g. [pitch_input, zdot_input]
    m : int
        Memory depth (same m as your Volterra kernel).
 
    Returns
    -------
    X : ndarray, shape (n_time, m, n_inputs)
        X[t, k, c] = inputs[c][t - (m-1) + k], causal and zero-padded at the
        start so the output has the SAME LENGTH n_time as your cl/cm arrays
        -- required to align 1:1 with your existing targets and with A @ h
        from the linear model if you use the residual hybrid below.
    """
    n_time = len(inputs[0])
    n_inputs = len(inputs)
    stacked = np.stack(inputs, axis=1)                      # (n_time, n_inputs)
    padded = np.vstack([np.zeros((m - 1, n_inputs)), stacked])
 
    X = np.empty((n_time, m, n_inputs), dtype=np.float64)
    for t in range(n_time):
        X[t] = padded[t : t + m]
    return X
 
 
class VolterraNNDataset(Dataset):
    """
    X : ndarray (n_time, m, n_inputs)
    y : ndarray (n_time, n_outputs)          raw targets, e.g. columns [cl, cm]
    residual_baseline : ndarray (n_time, n_outputs) or None
        If given, the NN is trained to predict y - residual_baseline instead
        of y directly -- this is the residual-on-Volterra hybrid: pass
        residual_baseline = np.stack([A @ h_cl, A @ h_cm], axis=1) using your
        existing linear kernels, and the NN only has to learn what the
        linear model misses.
    """
    def __init__(self, X, y, residual_baseline=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        target = y if residual_baseline is None else (y - residual_baseline)
        self.y = torch.tensor(target, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
 
# ============================================================================
# 2) TCN backbone
# ============================================================================
 
class CausalConv1d(nn.Module):
    """1D conv with left-only (causal) padding, so output[t] only depends on
    input[<=t] -- never looks into the future, matching the Volterra
    kernel's causality."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=self.pad, dilation=dilation)
        )
 
    def forward(self, x):
        out = self.conv(x)
        return out[..., : -self.pad] if self.pad != 0 else out
 
 
class TCNBlock(nn.Module):
    """
    Two causal convs + residual connection, standard TCN residual block.
 
    Includes a GroupNorm(num_groups=1, ...) -- i.e. a channel-wise layer
    norm -- after each conv, before the activation. Without this, a deep
    stack of residual-added ReLU blocks has no mechanism keeping
    activation scale bounded, and gradients/activations can blow up to
    inf/nan within a handful of training steps (exactly the failure you
    hit). GroupNorm(1, C) works for any channel count (unlike BatchNorm,
    which needs a large-enough batch to estimate stable statistics) and
    is a standard choice for TCNs for this reason.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm2 = nn.GroupNorm(1, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
 
    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)
 
 
class TCNBackbone(nn.Module):
    """
    Stack of TCNBlocks with exponentially increasing dilation (1, 2, 4, ...),
    so the receptive field grows fast with depth. Make sure the receptive
    field covers your full memory depth m -- see compute_receptive_field().
    """
    def __init__(self, n_inputs, channels=(16, 32, 32), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = n_inputs
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = in_ch
 
    def forward(self, x):
        # x: (batch, m, n_inputs) -> conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)
        out = self.net(x)           # (batch, out_channels, m)
        return out[:, :, -1]        # most recent causal timestep, one feature vector per window
 
 
def compute_receptive_field(kernel_size, n_layers):
    """
    Receptive field of a dilated-TCN stack with dilations 1,2,4,...,2^(L-1):
        RF = 1 + (kernel_size - 1) * (2^n_layers - 1)
    Pick channels/n_layers so RF >= m (your Volterra memory depth), or the
    network structurally cannot see far enough back to match the kernel.
    """
    return 1 + (kernel_size - 1) * (2 ** n_layers - 1)
 
 
# ============================================================================
# 3) Multi-head model
# ============================================================================
 
class VolterraNN(nn.Module):
    """
    Shared TCN trunk + one small head per output. Starting point:
    output_names=['cl', 'cm']. To extend to modes later:
        output_names=['cl', 'cm', 'mode_1', ..., 'mode_r']
    with no other code changes needed elsewhere.
    """
    def __init__(self, n_inputs, output_names,
                 channels=(16, 32, 32), kernel_size=3, dropout=0.1, head_hidden=16):
        super().__init__()
        self.output_names = list(output_names)
        self.backbone = TCNBackbone(n_inputs, channels, kernel_size, dropout)
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.backbone.out_channels, head_hidden),
                nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )
            for name in self.output_names
        })
 
    def forward(self, x):
        z = self.backbone(x)                                   # (batch, out_channels)
        outs = [self.heads[name](z) for name in self.output_names]
        return torch.cat(outs, dim=1)                           # (batch, n_outputs)
 
 
# ============================================================================
# 4) Variance-normalized loss (same fix as the OpenMDAO joint-objective issue)
# ============================================================================
 
def variance_normalized_mse(pred, target, var, eps=1e-12):
    """
    var : tensor, shape (n_outputs,)
        Per-output target variance, precomputed ONCE from the training set
        (do not recompute per batch). Dividing each output's squared error
        by its own variance puts cl and cm (or any future low-amplitude
        mode) on equal footing in the combined loss -- exactly the fix we
        needed for the joint SLSQP objective, now applied to NN training.
 
    eps guards against a near-zero variance (e.g. a constant or
    degenerate output column) causing a divide-by-zero -> inf -> nan.
    """
    err2 = (pred - target) ** 2
    return (err2 / var.clamp_min(eps)).mean()
 
 
# ============================================================================
# 5) Training loop
# ============================================================================
 
def train_volterra_nn(model, train_loader, val_loader, var,
                       epochs=300, lr=1e-4, weight_decay=2e-4, patience=25, # weight_decay=1e-4
                       device='cpu', verbose=True, grad_clip=1.0):
    """
    grad_clip : float or None
        Max gradient norm (torch.nn.utils.clip_grad_norm_). Unnormalized
        deep conv stacks (no BatchNorm/LayerNorm here) can develop
        exploding gradients within a handful of steps, especially at
        lr=1e-3; clipping is the cheapest safeguard. Set to None to
        disable, e.g. once you've confirmed training is stable and want to
        double check clipping isn't masking something else.
    """
    model.to(device)
    var = var.to(device)
    assert torch.isfinite(var).all(), f"Non-finite variance passed to loss: {var}"
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
 
    best_val = np.inf
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
 
    for epoch in range(epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
 
            if not torch.isfinite(xb).all() or not torch.isfinite(yb).all():
                raise ValueError(
                    f"Non-finite value in inputs/targets at epoch {epoch}, "
                    f"batch {batch_idx} -- check your raw data (pitch_input, "
                    f"zdot_input, cl, cm) for nan/inf BEFORE training, e.g.:\n"
                    f"  print(np.isnan(pitch_input).any(), np.isinf(pitch_input).any())\n"
                    f"  print(np.isnan(zdot_input).any(), np.isinf(zdot_input).any())\n"
                    f"  print(np.isnan(cl).any(), np.isnan(cm).any())"
                )
 
            optimizer.zero_grad()
            pred = model(xb)
            loss = variance_normalized_mse(pred, yb, var)
 
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_idx} "
                    f"(loss={loss.item()}). Likely exploding activations/gradients -- "
                    f"try a smaller lr (e.g. 1e-4), keep grad_clip enabled, or reduce "
                    f"channel width/depth to isolate whether the architecture is the cause."
                )
 
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
 
            train_loss += loss.item() * len(xb)
            n_train += len(xb)
        train_loss /= n_train
 
        model.eval()
        val_loss, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += variance_normalized_mse(model(xb), yb, var).item() * len(xb)
                n += len(xb)
        val_loss /= n
        scheduler.step(val_loss)
 
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
 
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val={best_val:.6e})")
                break
 
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:4d}  val_loss={val_loss:.6e}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
 
    model.load_state_dict(best_state)
    return model, best_val, history
 
def plot_training_curves(history, ax=None):
    """
    history : dict with 'train_loss', 'val_loss' (lists), as returned by
              train_volterra_nn.
    Log scale is used since the variance-normalized loss typically spans
    orders of magnitude over training.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(history['train_loss'], label='train')
    ax.semilogy(history['val_loss'], label='val')
    ax.set_xlabel('epoch')
    ax.set_ylabel('variance-normalized loss')
    ax.set_title('Training curves')
    ax.legend()
    return ax
 
 
def plot_time_series_comparison(pred, target, t, output_names,
                                  baseline=None, t_split=None, ncols=1):
    """
    Predicted vs actual time series, one subplot per output.
 
    baseline : ndarray, shape (n_time, n_outputs), or None
        Optional linear Volterra prediction (e.g. A @ h per output) to
        plot alongside the NN prediction and the actual data -- lets you
        see at a glance where the NN is correcting the linear fit versus
        where it's just tracking it.
    t_split : float or None
        If given, draws a vertical line (e.g. at the train/val boundary,
        or where a held-out test segment starts) so you can see whether
        accuracy degrades outside the trained region.
    """
    n_outputs = pred.shape[1]
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 2.5 * nrows),
                              sharex=True, squeeze=False)
 
    for i, name in enumerate(output_names):
        ax = axes[i // ncols, i % ncols]
        ax.plot(t, target[:, i], label='actual', color='k', linewidth=1)
        ax.plot(t, pred[:, i], label='predicted', color='tab:red',
                 linewidth=1, alpha=0.8)
        if baseline is not None:
            ax.plot(t, baseline[:, i], label='linear Volterra', color='tab:blue',
                     linewidth=1, alpha=0.6, linestyle='--')
        if t_split is not None:
            ax.axvline(t_split, color='gray', linestyle='--', alpha=0.6)
        ax.set_ylabel(name)
        ax.legend(loc='upper right', fontsize=8)
 
    axes[-1, 0].set_xlabel('time')
    fig.suptitle('Predicted vs actual')
    fig.tight_layout()
    return fig
 
 
def plot_parity(pred, target, output_names, ncols=2):
    """
    Predicted vs actual scatter (parity plot) per output, with a y=x
    reference line. Points hugging the diagonal indicate a good fit;
    systematic curvature indicates unmodeled nonlinearity or bias.
    """
    n_outputs = pred.shape[1]
    ncols = min(ncols, n_outputs)
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                              squeeze=False)
 
    for i, name in enumerate(output_names):
        ax = axes[i // ncols, i % ncols]
        ax.scatter(target[:, i], pred[:, i], s=4, alpha=0.3)
        lims = [min(target[:, i].min(), pred[:, i].min()),
                max(target[:, i].max(), pred[:, i].max())]
        ax.plot(lims, lims, 'k--', linewidth=1)
        ax.set_xlabel(f'actual {name}')
        ax.set_ylabel(f'predicted {name}')
        ax.set_title(name)
        ax.set_aspect('equal')
 
    for j in range(n_outputs, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')
 
    fig.tight_layout()
    return fig
 
 
def plot_nmse_bar(nmse_dict, ax=None):
    """
    Bar chart of NMSE per output/model, for comparing e.g. NN vs linear
    Volterra vs NN-hybrid side by side.
 
    nmse_dict : dict[str, dict[str, float]]
        Outer key = model name (e.g. 'linear', 'nn_from_scratch',
        'nn_hybrid'), inner dict = {output_name: nmse}.
        Example:
            {'linear':    {'cl': 0.08, 'cm': 0.15},
             'nn_hybrid': {'cl': 0.03, 'cm': 0.09}}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
 
    model_names = list(nmse_dict.keys())
    output_names = list(nmse_dict[model_names[0]].keys())
    n_models = len(model_names)
    n_outputs = len(output_names)
 
    x = np.arange(n_outputs)
    width = 0.8 / n_models
 
    for i, model_name in enumerate(model_names):
        values = [nmse_dict[model_name][o] for o in output_names]
        ax.bar(x + i * width, values, width, label=model_name)
 
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(output_names)
    ax.set_ylabel('NMSE')
    ax.set_yscale('log')
    ax.set_title('Model comparison')
    ax.legend()
    return ax

# ============================================================================
# 6) Prediction / evaluation helper
# ============================================================================
 
def predict(model, X, device='cpu'):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(xb).cpu().numpy()
    return pred
 
 
def per_output_nmse(pred, target):
    """NMSE per output column -- same diagnostic used throughout the Volterra work."""
    return np.mean((pred - target) ** 2, axis=0) / np.var(target, axis=0)

def solve_full_regularized(A, ys, m,
                            lambda_ridge=1e-3,
                            lambda_smooth=1e-2,
                            lambda_decay=1e-1,
                            decay_rate=5.0):
    """
    Closed-form solve of:
 
        min_h ||A h - y||^2
              + lambda_ridge   * ||h||^2
              + lambda_smooth  * ||D2 h||^2
              + lambda_decay   * h.T @ L @ h
 
    i.e. the exact convex problem your VoltterraResidualComp +
    RegularizationComp encode, but solved directly instead of via SLSQP.
    The normal-form system matrix
 
        M = A.T @ A + lambda_ridge*I + lambda_smooth*D2.T@D2 + lambda_decay*L
 
    is shared by every kernel/mode (only the right-hand side A.T @ y
    changes), so we factor it ONCE (Cholesky) and reuse that factorization
    for every column of `ys` -- this is the generalization of "compute the
    SVD once" to the full regularization suite, and is both faster and
    exact (no optimizer convergence to worry about).
 
    Note: uses normal equations (A.T @ A), so if A is very ill-conditioned
    consider passing a pre-whitened/pre-conditioned A, or fall back to
    solve_ridge_svd for the ridge-only case, which avoids squaring the
    condition number.
 
    Parameters
    ----------
    A : ndarray, shape (n_samples, n_coeffs)
    ys : ndarray, shape (n_samples,) or (n_samples, n_targets)
    m : int
        Memory length per kernel order (needed to build D2, L consistently
        with RegularizationComp).
 
    Returns
    -------
    h : ndarray, shape (n_coeffs,) or (n_coeffs, n_targets)
    """
    n_samples, n_coeffs = A.shape
    D2, L = build_regularization_matrices(n_coeffs, m, decay_rate=decay_rate)
 
    M = (A.T @ A
         + lambda_ridge * np.eye(n_coeffs)
         + lambda_smooth * (D2.T @ D2)
         + lambda_decay * L)
 
    rhs = A.T @ ys  # shape (n_coeffs,) or (n_coeffs, n_targets)
 
    c, low = cho_factor(M)
    h = cho_solve((c, low), rhs)
    return h


def build_regularization_matrices(n_coeffs, m, decay_rate=5.0):
    """
    Same D2 (smoothness) and L (decay) matrices as RegularizationComp,
    so results are directly comparable to the OpenMDAO pipeline.
    """
    n_orders = n_coeffs // m
 
    D2_single = np.diff(np.eye(m), n=2, axis=0)
    L_single = np.diag(np.exp(decay_rate * np.linspace(0, 1, m)) - 1)
 
    D2 = block_diag(*[D2_single for _ in range(n_orders)])
    L = block_diag(*[L_single for _ in range(n_orders)])
 
    return D2, L

# ============================================================================
# Example end-to-end usage
# ============================================================================
if __name__ == '__main__':

    # device 
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ------------------------------------------------------------------
    # Data: signals + targets
    # ------------------------------------------------------------------
      
    AOA = 5
    MACH = 80
    dt = 2.5e-4

    chord = 0.4064 #1. 
    v_inf = 123. * MACH / 80 # * 277.24 # 123. 

    home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine" % AOA

    filename = home_pitch + "/history_05_00002.dat"    

    chord = 0.4064 #1. 
    v_inf = 123. * MACH / 80 # * 277.24 # 123. 
    print("V freestream", v_inf)

    tref = chord / (2 * v_inf)
    subs = 48 

    dt = 2.5e-4 

    iCd = 11
    iCl = 12
    iCm = 16

    print(filename)
    data = np.array(read_csv(filename)) 
    cd = (data[0*subs::subs,iCd] ) #/ daoa_plunge
    cl_output = (data[0*subs::subs,iCl] ) #/ daoa_plunge
    cm_output = - (data[0::subs,iCm] ) # sign differs in 2 and 3D?
    it_05 = data[0*subs::subs,0]
    tau_vector = (it_05-2)*dt/tref
    tt_05 = (it_05-2)*dt

    print(cl_output.shape,it_05.shape,it_05[0],it_05[-1])

    ntsteps = len(cl_output)

    target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.849, 3.6304, 5.30]) 
    target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226, 3.43, 5.436])

    t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt,
                                                   ntsteps*dt, amp=0.95*np.pi/180.)
    t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt,
                                                    ntsteps*dt, amp=0.065)
    zdot_input = np.gradient(plunge_input, t) / v_inf

    cl = cl_output - np.mean(cl_output)
    cm = cm_output - np.mean(cm_output)

    m = 72                              # SAME memory depth as your Volterra fit

    A_pitch = np.zeros((ntsteps,m))
    A_zdot = np.zeros((ntsteps,m))

    for icolumn in range(m):
        A_pitch[icolumn:,icolumn] = pitch_input[:ntsteps-icolumn] 
        A_zdot[icolumn:,icolumn] = zdot_input[:ntsteps-icolumn] 

    A_123 = np.hstack( (A_pitch, A_zdot))

    X = build_lagged_windows([pitch_input, zdot_input], m)   # (n_time, m, 2)
    y = np.stack([cl, cm], axis=1)                            # (n_time, 2)
    
    # ------------------------------------------------------------------
    # Optional: residual-on-Volterra hybrid
    # ------------------------------------------------------------------
    # from regularized_mode_solve import solve_full_regularized
    h_cl = solve_full_regularized(A_123, cl, m, lambda_ridge=1e-3,
                                   lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    h_cm = solve_full_regularized(A_123, cm, m, lambda_ridge=1e-3,
                                   lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    baseline = np.stack([A_123 @ h_cl, A_123 @ h_cm], axis=1)   # (n_time, 2)

    plt.figure(figsize=(12, 9))

    plt.subplot(2,1,1)

    plt.plot(A_123@ h_cl, 'b-', linewidth=1,
             label='Volterra')

    plt.plot(cl, 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    # plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    plt.subplot(2,1,2)

    plt.plot(A_123@ h_cm, 'b-', linewidth=1,
             label='Volterra')

    plt.plot(cm, 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    # plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    plt.show()

    dataset = VolterraNNDataset(X, y, residual_baseline=baseline)
    
    # ------------------------------------------------------------------
    # Or, train from scratch (no linear baseline):
    # ------------------------------------------------------------------
    # dataset = VolterraNNDataset(X, y)
    #
    # ------------------------------------------------------------------
    # Split, wrap in DataLoaders
    # ------------------------------------------------------------------
    n_val = int(0.2 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # ------------------------------------------------------------------
    # Variance for the loss -- computed ONCE from the training targets
    # ------------------------------------------------------------------
    train_targets = dataset.y[train_ds.indices].numpy()
    var = torch.tensor(train_targets.var(axis=0), dtype=torch.float32)
    
    # ------------------------------------------------------------------
    # Build model; check receptive field covers m
    # ------------------------------------------------------------------
    channels = (16, 32, 32, 32)   # 4 layers -> dilations 1,2,4,8
    rf = compute_receptive_field(kernel_size=3, n_layers=len(channels))
    print(f"receptive field = {rf}, memory depth m = {m}")  # want rf >= m
    
    model = VolterraNN(n_inputs=2, output_names=['cl', 'cm'], channels=channels)
    
    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    # model, best_val, history = train_volterra_nn(model, train_loader, val_loader, var,
    #                                     epochs=300, lr=1e-3)

    model, best_val, history = train_volterra_nn(model, train_loader, val_loader, var, device=device,epochs=300, lr=1e-3)
    plot_training_curves(history)

    # ------------------------------------------------------------------
    # Evaluate (remember to add the linear baseline back if using the hybrid)
    # ------------------------------------------------------------------
    # pred = predict(model, X)
    pred = predict(model, X, device=device)
    # if using hybrid: 
    pred = pred + baseline
    print(per_output_nmse(pred, y))

    output_names = ['cl', 'cm']

    plot_time_series_comparison(pred, y, t, output_names, baseline=baseline)
    plot_parity(pred, y, output_names)
    
    nn_nmse = dict(zip(output_names, per_output_nmse(pred, y)))
    linear_pred = baseline  # A @ h, no NN correction
    linear_nmse = dict(zip(output_names, per_output_nmse(linear_pred, y)))
    
    plot_nmse_bar({'linear': linear_nmse, 'nn_hybrid': nn_nmse})
    
    plt.show()

    save_path = home_pitch +"/volterra_nn_checkpoint.pt"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'output_names': model.output_names,
        'n_inputs': 2,                      # pitch, zdot
        'channels': channels,               # e.g. (16, 32, 32, 32) -- whatever you used
        'kernel_size': 3,
        'm': m,                             # memory depth -- needed to rebuild windows later
        'var': var,                         # the loss-normalization variance -- keep for reference
    }
    torch.save(checkpoint, save_path)
 