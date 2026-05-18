# Rough SPDE Engine

Implements a fractional stochastic partial differential equation (SPDE) driven by fractional Brownian motion (Hurst=0.3) to model ETF return fields. The linear operator L is approximated by a neural operator (1D CNN). The model is trained on spatial fields interpolated from ETF returns and predicts the next day's return field. Multi‑window evaluation selects the best window per ETF.

- **SPDE:** du = (L u) dt + dW^H(t,x)
- **Noise:** Fractional Brownian motion (Hurst=0.3)
- **Operator:** Neural operator (CNN)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by predicted return

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
