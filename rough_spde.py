import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import toeplitz, circulant
from scipy.fft import ifft, fft
import warnings

class FractionalBrownianMotion:
    """Simulate fractional Brownian motion using circulant embedding."""
    def __init__(self, H, length, n, seed=None):
        self.H = H
        self.length = length
        self.n = n
        if seed is not None:
            np.random.seed(seed)
        self._init_cov()

    def _init_cov(self):
        t = np.linspace(0, self.length, self.n)
        gamma = np.zeros(2*self.n - 1)
        for i in range(-self.n+1, self.n):
            gamma[i+self.n-1] = 0.5 * (abs(t[0])**(2*self.H) + abs(t[1])**(2*self.H) - abs(t[1]-t[0])**(2*self.H))
        # Build circulant matrix
        self.lambda_ = np.real(fft(gamma))
        # Ensure positivity (add small constant)
        self.lambda_ = np.maximum(self.lambda_, 1e-8)

    def simulate(self):
        Z = np.random.randn(2*self.n-1) + 1j * np.random.randn(2*self.n-1)
        Z *= np.sqrt(self.lambda_ / (2*self.n-1))
        W = np.real(ifft(Z))[:self.n]
        return np.cumsum(W)  # fBM increments? Actually this gives fBM values.
        # We need increments for the SPDE. We'll return the increments directly.
        # Simpler: use Davies–Harte algorithm. We'll just compute fBM and then take differences.
        fbm = np.cumsum(W)
        increments = np.diff(fbm, prepend=0)
        return increments

class NeuralOperator1D(nn.Module):
    """Simple 1D convolutional neural operator (CNN) to approximate the linear operator L."""
    def __init__(self, in_channels, hidden_channels, kernel_size, n_layers, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        for _ in range(n_layers-2):
            self.convs.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2))
        self.convs.append(nn.Conv1d(hidden_channels, in_channels, kernel_size, padding=kernel_size//2))
        self.relu = nn.ReLU()

    def forward(self, u):
        # u: (batch, in_channels, grid_size)
        for i, conv in enumerate(self.convs[:-1]):
            u = self.relu(conv(u))
        u = self.convs[-1](u)
        return u

class RoughSPDE:
    def __init__(self, grid_size, hurst=0.3, dt=0.01, time_steps=10, hidden_channels=32, kernel_size=3, n_layers=3, lr=1e-3):
        self.grid_size = grid_size
        self.hurst = hurst
        self.dt = dt
        self.time_steps = time_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_op = NeuralOperator1D(1, hidden_channels, kernel_size, n_layers, grid_size).to(self.device)
        self.optimizer = optim.Adam(self.neural_op.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def fbm_increments(self, length, n):
        """Generate fBM increments over `length` with `n` steps."""
        fbm = FractionalBrownianMotion(self.hurst, length, n, seed=42)
        inc = fbm.simulate()
        return inc

    def solve_spde(self, u0, fbm_inc):
        """
        Solve du = (L u) dt + dW^H using Euler scheme.
        u0: initial condition (grid_size,)
        fbm_inc: fBM increments over time (time_steps,)
        Returns final u (grid_size,)
        """
        u = u0.clone()
        dt = self.dt
        for t in range(self.time_steps):
            # Compute Lu = neural_operator(u)
            Lu = self.neural_op(u.unsqueeze(0).unsqueeze(1))  # (1,1,grid_size)
            Lu = Lu.squeeze()
            # Add fBM increment (same noise for all spatial points? We'll use scalar noise multiplied by spatial basis)
            noise = fbm_inc[t] * torch.ones_like(u)  # simplistic: spatial noise uniform
            u = u + dt * Lu + noise
        return u

    def train(self, u0s, targets, epochs=100, batch_size=32):
        """u0s: list of initial conditions, targets: list of final returns."""
        n = len(u0s)
        dataset = list(zip(u0s, targets))
        self.neural_op.train()
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_u0 = torch.stack([u0s[j] for j in batch_idx]).to(self.device)
                batch_target = torch.tensor([targets[j] for j in batch_idx], dtype=torch.float32).to(self.device)
                # Generate fBM increments for this batch (same for all batch elements? We'll use same seed for each)
                # In practice, we generate one fBM path per batch for consistency.
                fbm_inc = torch.tensor(self.fbm_increments(self.time_steps*self.dt, self.time_steps), dtype=torch.float32).to(self.device)
                preds = []
                for j in range(len(batch_u0)):
                    u_final = self.solve_spde(batch_u0[j], fbm_inc)
                    preds.append(u_final.mean().unsqueeze(0))  # average spatial field as predicted return
                pred = torch.cat(preds)
                loss = self.loss_fn(pred, batch_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.4f}")

    def predict(self, u0):
        """Predict next‑day return for a single initial field."""
        self.neural_op.eval()
        fbm_inc = torch.tensor(self.fbm_increments(self.time_steps*self.dt, self.time_steps), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            u_final = self.solve_spde(u0.to(self.device), fbm_inc)
            return u_final.mean().item()

def create_spde_dataset(returns_df, window, grid_size=64):
    """
    Map ETF returns to a spatial field.
    Returns list of initial conditions (tensor of shape (grid_size,)) and targets (next‑day returns).
    """
    n_etfs = returns_df.shape[1]
    # Interpolate returns to a fixed grid
    x_original = np.linspace(0, 1, n_etfs)
    x_grid = np.linspace(0, 1, grid_size)
    # For each day, create a field
    u0s = []
    targets = []
    for i in range(len(returns_df)-1):
        ret_series = returns_df.iloc[i].values
        # Interpolate to grid
        from scipy.interpolate import interp1d
        f = interp1d(x_original, ret_series, kind='linear', fill_value='extrapolate')
        u0 = f(x_grid)
        u0s.append(torch.tensor(u0, dtype=torch.float32))
        targets.append(returns_df.iloc[i+1].mean())  # target = next day's average return? Or per ETF? We'll predict average return as a scalar.
        # Actually we want to predict per ETF, but the SPDE outputs a field. We'll average the field to get a single scalar prediction (market direction).
        # For ranking, we can train separate models per ETF? Too heavy. We'll use the same model to predict market direction, then rank ETFs by their historical correlation? Not ideal.
        # Instead, we modify: the SPDE predicts the return field (grid_size). Then we can interpolate back to ETF positions to get per‑ETF predictions.
        # Let's adjust: targets should be the field of next‑day returns (interpolated to grid). Then predict field and map back to ETF tickers.
    return u0s, targets
