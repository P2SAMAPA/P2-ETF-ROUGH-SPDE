import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from scipy.fft import fft, ifft

class FractionalBrownianMotion:
    """Fractional Brownian motion via circulant embedding (Davies–Harte)."""
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
            gamma[i+self.n-1] = 0.5 * (abs(t[0])**(2*self.H) + abs(t[0])**(2*self.H) - abs(t[0]-t[1])**(2*self.H))  # simplified
        # Build circulant covariance
        lambd = np.real(fft(gamma))
        self.lambd = np.maximum(lambd, 1e-8)

    def simulate(self):
        Z = np.random.randn(2*self.n-1) + 1j * np.random.randn(2*self.n-1)
        Z *= np.sqrt(self.lambd / (2*self.n-1))
        W = np.real(ifft(Z))[:self.n]
        return np.cumsum(W)  # fBM values
        # increments = np.diff(fbm, prepend=0)
        # return increments

class NeuralOperator1D(nn.Module):
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
        for i, conv in enumerate(self.convs[:-1]):
            u = self.relu(conv(u))
        u = self.convs[-1](u)
        return u

class RoughSPDE:
    def __init__(self, grid_size, hurst=0.3, dt=0.01, time_steps=10,
                 hidden_channels=32, kernel_size=3, n_layers=3, lr=1e-3):
        self.grid_size = grid_size
        self.hurst = hurst
        self.dt = dt
        self.time_steps = time_steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_op = NeuralOperator1D(1, hidden_channels, kernel_size, n_layers, grid_size).to(self.device)
        self.optimizer = optim.Adam(self.neural_op.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def fbm_increments(self):
        fbm = FractionalBrownianMotion(self.hurst, self.time_steps*self.dt, self.time_steps+1, seed=42)
        path = fbm.simulate()
        increments = np.diff(path)
        return torch.tensor(increments, dtype=torch.float32).to(self.device)

    def solve_spde(self, u0, noise):
        u = u0.clone()
        dt = self.dt
        for t in range(self.time_steps):
            Lu = self.neural_op(u.unsqueeze(0).unsqueeze(1)).squeeze()
            u = u + dt * Lu + noise[t] * torch.ones_like(u)
        return u

    def train(self, u0_list, target_field_list, epochs=50, batch_size=32):
        n = len(u0_list)
        indices = np.arange(n)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            total_loss = 0.0
            for i in range(0, n, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_u0 = torch.stack([u0_list[j] for j in batch_idx]).to(self.device)
                batch_target = torch.stack([target_field_list[j] for j in batch_idx]).to(self.device)
                # Generate one noise path for the batch (same for all in batch)
                noise = self.fbm_increments()
                preds = []
                for j in range(len(batch_u0)):
                    u_final = self.solve_spde(batch_u0[j], noise)
                    preds.append(u_final)
                pred = torch.stack(preds)
                loss = self.loss_fn(pred, batch_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/len(indices):.6f}")

    def predict(self, u0):
        self.neural_op.eval()
        noise = self.fbm_increments()
        with torch.no_grad():
            u_final = self.solve_spde(u0.to(self.device), noise)
            return u_final.cpu().numpy()

def create_spde_dataset(returns_df, window, grid_size=64):
    """Create spatial field from returns."""
    n_etfs = returns_df.shape[1]
    etf_names = returns_df.columns.tolist()
    x_original = np.linspace(0, 1, n_etfs)
    x_grid = np.linspace(0, 1, grid_size)
    u0_list = []
    target_field_list = []
    for i in range(len(returns_df)-1):
        ret_series = returns_df.iloc[i].values
        # Interpolate to grid
        f = interp1d(x_original, ret_series, kind='linear', fill_value='extrapolate')
        u0 = f(x_grid)
        u0_list.append(torch.tensor(u0, dtype=torch.float32))
        # Target: next day returns interpolated to grid
        next_ret = returns_df.iloc[i+1].values
        f_target = interp1d(x_original, next_ret, kind='linear', fill_value='extrapolate')
        target = f_target(x_grid)
        target_field_list.append(torch.tensor(target, dtype=torch.float32))
    return u0_list, target_field_list, x_original, etf_names

def interpolate_field_to_etfs(field, x_original, x_grid, etf_names):
    """Interpolate field values back to ETF positions."""
    from scipy.interpolate import interp1d
    f = interp1d(x_grid, field, kind='linear', fill_value='extrapolate')
    values = f(x_original)
    return {etf_names[i]: values[i] for i in range(len(etf_names))}
