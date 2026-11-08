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
from torch.utils.data import TensorDataset,Dataset, DataLoader, random_split
from volterra_nn_v0 import (
     solve_full_regularized, plot_time_series_comparison,
     plot_parity, plot_nmse_bar, read_csv, generate_schroeder_multisine, plot_training_curves
#     build_lagged_windows, per_output_nmse, predict,
)
import matplotlib.pyplot as plt


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
 
 
def build_multi_signal_dataset(signals_list, m, residual_baselines=None):
    """
    Build lagged windows + targets from SEVERAL distinct excitation
    realizations, concatenated into one dataset -- use this instead of
    training on a single signal, to avoid the network overfitting to one
    particular multisine's phase/frequency content.
 
    IMPORTANT: windows are built SEPARATELY per signal, then concatenated
    -- never build windows on a naive concatenation of raw signals, or
    windows near each boundary would wrongly mix the end of one signal
    with the start of the next.
 
    Parameters
    ----------
    signals_list : list of dict, one per signal/realization, each with:
        {'inputs': [pitch_input, zdot_input], 'targets': [cl, cm]}
    m : int
        Memory depth (same m used everywhere else).
    residual_baselines : list of ndarray, shape (n_time_i, n_outputs), or None
        One per signal, if using the residual hybrid (A_i @ h computed with
        each signal's own design matrix but your SAME fitted h).
 
    Returns
    -------
    X : ndarray, shape (sum(n_time_i), m, n_inputs)
    y : ndarray, shape (sum(n_time_i), n_outputs)
    signal_id : ndarray, shape (sum(n_time_i),)
        Which original signal each row came from -- use this to do a
        SIGNAL-LEVEL train/val split (see split_by_signal below), instead
        of a random per-window split that leaks across overlapping windows.
    baseline : ndarray or None, same shape as y
    """
    X_list, y_list, id_list, baseline_list = [], [], [], []
 
    for i, sig in enumerate(signals_list):
        X_i = build_lagged_windows(sig['inputs'], m)
        y_i = np.stack(sig['targets'], axis=1)
        X_list.append(X_i)
        y_list.append(y_i)
        id_list.append(np.full(len(y_i), i))
        if residual_baselines is not None:
            baseline_list.append(residual_baselines[i])
 
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    signal_id = np.concatenate(id_list, axis=0)
    baseline = np.concatenate(baseline_list, axis=0) if residual_baselines is not None else None
 
    return X, y, signal_id, baseline
 
 
def chronological_split_multi_signal(signal_id, val_fraction=0.15, test_fraction=0.15, purge=0):
    """
    Alternative to split_by_signal(), for when you don't yet have enough
    DISTINCT signals to hold one out entirely. Splits each signal
    CHRONOLOGICALLY -- the last (val_fraction + test_fraction) of each
    signal's timeline is held out, the rest goes to training -- rather
    than holding out a whole signal.
 
    This still combines all signals' early portions for training
    diversity, and avoids the random-window leakage problem (adjacent,
    near-duplicate windows split across train/val). It is a WEAKER
    generalization test than holding out a whole distinct signal (see
    split_by_signal): it checks generalization to a later time segment of
    the SAME excitation realizations, not to genuinely new excitation
    content. Switch to split_by_signal once a third signal is available.
 
    Relies on rows for each signal being contiguous and time-ordered in
    X/y/signal_id, which build_multi_signal_dataset guarantees.
 
    purge : int
        Number of rows to drop right before the val/test boundary within
        each signal, to remove the (minor) boundary leakage from windows
        whose lagged inputs straddle the split point. 0 is usually fine
        given m is typically small relative to signal length; increase if
        you want a stricter cut.
 
    Returns
    -------
    train_idx, val_idx, test_idx : ndarray of row indices into X/y
    """
    train_idx_list, val_idx_list, test_idx_list = [], [], []
 
    for sid in np.unique(signal_id):
        idx = np.where(signal_id == sid)[0]  # contiguous, time-ordered
        n = len(idx)
        n_val = int(n * val_fraction)
        n_test = int(n * test_fraction)
        val_start = n - n_val - n_test
        test_start = n - n_test
 
        train_idx_list.append(idx[: max(val_start - purge, 0)])
        val_idx_list.append(idx[val_start:test_start])
        test_idx_list.append(idx[test_start:])
 
    return (np.concatenate(train_idx_list),
            np.concatenate(val_idx_list),
            np.concatenate(test_idx_list))
    """
    Build train/val index arrays by WHICH SIGNAL each window came from,
    rather than randomly across all windows -- avoids the leakage problem
    described above (adjacent, near-duplicate windows split across
    train/val). Hold out one or more entire signals for validation.
 
    val_signal_ids : list/set of ints
        Which signal indices (matching signals_list order) go to
        validation. Everything else goes to train.
    """
    val_mask = np.isin(signal_id, list(val_signal_ids))
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx
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
 
def output_smoothness_penalty(pred_sequence):
    """
    Second-difference roughness penalty on a CHRONOLOGICALLY ORDERED batch
    of predictions -- the NN-output analogue of RegularizationComp's
    lambda_smooth * ||D2 @ h||^2 term, but applied to the predicted time
    series instead of the kernel coefficients. Only meaningful if
    pred_sequence's rows are in time order (a shuffled batch has no
    temporal structure to penalize).
 
    pred_sequence : tensor, shape (seq_len, n_outputs)
    """
    d2 = pred_sequence[2:] - 2 * pred_sequence[1:-1] + pred_sequence[:-2]
    return (d2 ** 2).mean()
 
 
def train_volterra_nn(model, train_loader, val_loader, var, X_full=None,
                       epochs=300, lr=1e-4, weight_decay=1e-4, patience=25,
                       device='cpu', verbose=True, grad_clip=1.0,
                       lambda_smooth_output=0.0, smooth_chunk_len=256):
    """
    X_full : ndarray, shape (n_time, m, n_inputs), or None
        The FULL, chronologically-ordered lagged-window array (same X you
        pass to build the shuffled train/val split). Required only if
        lambda_smooth_output > 0.
    lambda_smooth_output : float
        Weight on output_smoothness_penalty. 0 (default) disables it --
        existing training runs are unaffected unless you opt in. Start
        small (e.g. 1e-3 to 1e-1 relative to your main loss scale) and
        increase if high-frequency artifacts persist; too large will
        smear out genuine fast dynamics, same trade-off as lambda_smooth
        did for the linear kernel.
    smooth_chunk_len : int
        Length of the random contiguous, time-ordered chunk drawn each
        step to evaluate the smoothness penalty on.
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
 
    X_full_t = None
    if lambda_smooth_output > 0:
        assert X_full is not None, "X_full is required when lambda_smooth_output > 0"
        X_full_t = torch.tensor(X_full, dtype=torch.float32).to(device)
        n_time_full = X_full_t.shape[0]
 
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
 
            if lambda_smooth_output > 0:
                # Draw a random contiguous, chronologically-ordered chunk
                # from the FULL series (independent of this shuffled
                # batch) and penalize its second-difference roughness.
                start = np.random.randint(0, n_time_full - smooth_chunk_len)
                chunk = X_full_t[start : start + smooth_chunk_len]
                pred_chunk = model(chunk)  # (smooth_chunk_len, n_outputs), time-ordered
                loss = loss + lambda_smooth_output * output_smoothness_penalty(pred_chunk)
 
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
 
 
# ============================================================================
# 6) Prediction / evaluation helper
# ============================================================================
 
def predict(model, X, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(xb).cpu().numpy()
    return pred
 
 
def per_output_nmse(pred, target):
    """NMSE per output column -- same diagnostic used throughout the Volterra work."""
    return np.mean((pred - target) ** 2, axis=0) / np.var(target, axis=0)
 
 


# ============================================================================
# Example end-to-end usage
# ============================================================================
if __name__ == '__main__':
    # device 
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ------------------------------------------------------------------
    # Data: signals 1 + targets
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

    t, pitch_input_1 = generate_schroeder_multisine(target_freqs_pitching, dt,
                                                   ntsteps*dt, amp=0.95*np.pi/180.)
    t, plunge_input_1 = generate_schroeder_multisine(target_freqs_plunging, dt,
                                                    ntsteps*dt, amp=0.065)
    zdot_input_1 = np.gradient(plunge_input_1, t) / v_inf

    cl_1 = cl_output - np.mean(cl_output)
    cm_1 = cm_output - np.mean(cm_output)

    m = 72                              # SAME memory depth as your Volterra fit

    A_pitch = np.zeros((ntsteps,m))
    A_zdot = np.zeros((ntsteps,m))

    for icolumn in range(m):
        A_pitch[icolumn:,icolumn] = pitch_input_1[:ntsteps-icolumn] 
        A_zdot[icolumn:,icolumn] = zdot_input_1[:ntsteps-icolumn] 

    A_123 = np.hstack( (A_pitch, A_zdot, A_pitch*A_pitch, A_pitch*A_zdot, A_zdot*A_zdot))
    
    # ------------------------------------------------------------------
    # Optional: residual-on-Volterra hybrid
    # ------------------------------------------------------------------
    # from regularized_mode_solve import solve_full_regularized
    h_cl_1 = solve_full_regularized(A_123, cl_1, m, lambda_ridge=1e-3,
                                   lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    h_cm_1 = solve_full_regularized(A_123, cm_1, m, lambda_ridge=1e-3,
                                   lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    baseline_1 = np.stack([A_123 @ h_cl_1, A_123 @ h_cm_1], axis=1)   # (n_time, 2)

    plt.figure(figsize=(12, 9))

    plt.subplot(2,1,1)

    plt.plot(A_123@ h_cl_1, 'b-', linewidth=1,
             label='Volterra')

    plt.plot(cl_1, 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    # plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    plt.subplot(2,1,2)

    plt.plot(A_123@ h_cm_1, 'b-', linewidth=1,
             label='Volterra')

    plt.plot(cm_1, 'k-', linewidth=1,
             label='Reference')

    plt.xlabel('t')
    # plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    plt.show()

    # ------------------------------------------------------------------
    # Data: signals 2 + targets
    # ------------------------------------------------------------------
      
    home_pitch = "/Users/marcello/Documents/SU2/SC20414/AOA0%s_R12_veryfine_2" % AOA

    filename = home_pitch + "/history_05_00002.dat"    

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

    target_freqs_pitching = 24.9 / (2.*np.pi) * np.array([1.0, 1.8849]) 
    target_freqs_plunging = 18.1 / (2.*np.pi) * np.array([1.0, 1.9226])

    t, pitch_input_2 = generate_schroeder_multisine(target_freqs_pitching, dt,
                                                    ntsteps*dt, amp=0.95*np.pi/180.)
    t, plunge_input_2 = generate_schroeder_multisine(target_freqs_plunging, dt,
                                                    ntsteps*dt, amp=0.065)
    zdot_input_2 = np.gradient(plunge_input_2, t) / v_inf

    cl_2 = cl_output - np.mean(cl_output)
    cm_2 = cm_output - np.mean(cm_output)

    m = 72                              # SAME memory depth as your Volterra fit

    A_pitch = np.zeros((ntsteps,m))
    A_zdot = np.zeros((ntsteps,m))

    for icolumn in range(m):
        A_pitch[icolumn:,icolumn] = pitch_input_2[:ntsteps-icolumn] 
        A_zdot[icolumn:,icolumn] = zdot_input_2[:ntsteps-icolumn] 

    A_123 = np.hstack( (A_pitch, A_zdot, A_pitch*A_pitch, A_pitch*A_zdot, A_zdot*A_zdot))

    # ------------------------------------------------------------------
    # Optional: residual-on-Volterra hybrid
    # ------------------------------------------------------------------
    # from regularized_mode_solve import solve_full_regularized
    h_cl_2 = solve_full_regularized(A_123, cl_2, m, lambda_ridge=1e-3,
                                    lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    h_cm_2 = solve_full_regularized(A_123, cm_2, m, lambda_ridge=1e-3,
                                    lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    baseline_2 = np.stack([A_123 @ h_cl_2, A_123 @ h_cm_2], axis=1)   # (n_time, 2)

    plt.figure(figsize=(12, 9))

    plt.subplot(2,1,1)

    plt.plot(A_123@ h_cl_2, 'b-', linewidth=1,
                label='Volterra')

    plt.plot(cl_2, 'k-', linewidth=1,
                label='Reference')

    plt.xlabel('t')
    # plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    plt.subplot(2,1,2)

    plt.plot(A_123@ h_cm_2, 'b-', linewidth=1,
                label='Volterra')

    plt.plot(cm_2, 'k-', linewidth=1,
                label='Reference')

    plt.xlabel('t')
    # plt.ylabel(f'q_{i+1}')
    plt.grid(True)

    plt.show()

    # --- combine both signals ---
    def augment_quadratic(pitch, zdot):
        return [pitch, zdot, pitch**2, zdot**2, pitch * zdot]

    signals_list = [
        {'inputs': augment_quadratic(pitch_input_1, zdot_input_1),
        'targets': [cl_1 - baseline_1[:, 0], cm_1 - baseline_1[:, 1]]},
        {'inputs': augment_quadratic(pitch_input_2, zdot_input_2),
        'targets': [cl_2 - baseline_2[:, 0], cm_2 - baseline_2[:, 1]]},
    ]
    X, y, signal_id, _ = build_multi_signal_dataset(signals_list, m)



    # --- chronological split: last 15% of EACH signal -> val, next 15% -> test, rest -> train ---
    train_idx, val_idx, test_idx = chronological_split_multi_signal(
        signal_id, val_fraction=0.15, test_fraction=0.15
    )

    def standardize_channels(X, train_idx):
        """z-score each input channel, using stats from TRAINING data only."""
        mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
        std = X[train_idx].std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1.0
        return (X - mean) / std, mean, std

    X_norm, ch_mean, ch_std = standardize_channels(X, train_idx)

    train_ds = TensorDataset(
        torch.tensor(X_norm[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_norm[val_idx], dtype=torch.float32),
        torch.tensor(y[val_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    var = torch.tensor(y[train_idx].var(axis=0), dtype=torch.float32)

    # --- training: exactly your existing call, unchanged ---
    channels = (16, 32, 32, 32)
    rf = compute_receptive_field(kernel_size=3, n_layers=len(channels))
    print(f"receptive field = {rf}, memory depth m = {m}")

    model = VolterraNN(n_inputs=5, output_names=['cl', 'cm'], channels=channels)

    model, best_val, history = train_volterra_nn(
        model, train_loader, val_loader, var, device=device, epochs=120, lr=2.5e-4
    )
    plot_training_curves(history)

    # --- final, honest evaluation on the held-out test segment ---
    pred_test = predict(model, X_norm[test_idx], device=device)
    test_nmse = dict(zip(['cl', 'cm'], per_output_nmse(pred_test, y[test_idx])))
    print("Test NMSE (held-out chronological segment, both signals):", test_nmse)




    # ------------------------------------------------------------------
    # Data: signals + targets
    # ------------------------------------------------------------------
    # t, pitch_input = generate_schroeder_multisine(target_freqs_pitching, dt,
    #                                                ntsteps*dt, amp=0.95*np.pi/180.)
    # t, plunge_input = generate_schroeder_multisine(target_freqs_plunging, dt,
    #                                                 ntsteps*dt, amp=0.065)
    # zdot_input = np.gradient(plunge_input, t) / v_inf
    #
    # cl, cm = load_cl_cm_from_file(...)   # shape (n_time,) each
    #
    # m = ...                              # SAME memory depth as your Volterra fit
    # X = build_lagged_windows([pitch_input, zdot_input], m)   # (n_time, m, 2)
    # y = np.stack([cl, cm], axis=1)                            # (n_time, 2)
    #
    # ------------------------------------------------------------------
    # Optional: residual-on-Volterra hybrid
    # ------------------------------------------------------------------
    # from regularized_mode_solve import solve_full_regularized
    # h_cl = solve_full_regularized(A_123, cl, m, lambda_ridge=1e-3,
    #                                lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    # h_cm = solve_full_regularized(A_123, cm, m, lambda_ridge=1e-3,
    #                                lambda_smooth=1e-2, lambda_decay=1e-1, decay_rate=5.0)
    # baseline = np.stack([A_123 @ h_cl, A_123 @ h_cm], axis=1)   # (n_time, 2)
    # dataset = VolterraNNDataset(X, y, residual_baseline=baseline)
    #
    # ------------------------------------------------------------------
    # Or, train from scratch (no linear baseline):
    # ------------------------------------------------------------------
    # dataset = VolterraNNDataset(X, y)
    #
    # ------------------------------------------------------------------
    # Split, wrap in DataLoaders
    # ------------------------------------------------------------------
    # n_val = int(0.2 * len(dataset))
    # train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    # train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    #
    # ------------------------------------------------------------------
    # Variance for the loss -- computed ONCE from the training targets
    # ------------------------------------------------------------------
    # train_targets = dataset.y[train_ds.indices].numpy()
    # var = torch.tensor(train_targets.var(axis=0), dtype=torch.float32)
    #
    # ------------------------------------------------------------------
    # Build model; check receptive field covers m
    # ------------------------------------------------------------------
    # channels = (16, 32, 32, 32)   # 4 layers -> dilations 1,2,4,8
    # rf = compute_receptive_field(kernel_size=3, n_layers=len(channels))
    # print(f"receptive field = {rf}, memory depth m = {m}")  # want rf >= m
    #
    # model = VolterraNN(n_inputs=2, output_names=['cl', 'cm'], channels=channels)
    #
    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    # model, best_val = train_volterra_nn(model, train_loader, val_loader, var,
    #                                      epochs=300, lr=1e-3)
    #
    # ------------------------------------------------------------------
    # Evaluate (remember to add the linear baseline back if using the hybrid)
    # ------------------------------------------------------------------
    # pred = predict(model, X)
    # if using hybrid: pred = pred + baseline
    # print(per_output_nmse(pred, y))
    pass

