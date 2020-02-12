import anmf
import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st
import torch.utils.data as td

def _simulate_pois():
  n = 500
  p = 256
  k = 3
  np.random.seed(0)
  l = np.random.lognormal(sigma=0.5, size=(n, k))
  f = np.random.lognormal(sigma=0.5, size=(p, k))
  lam = l @ f.T
  x = np.random.poisson(lam=lam)
  llik = st.poisson(mu=lam).logpmf(x).sum()
  return x, llik

@pytest.fixture
def simulate_pois():
  return _simulate_pois()

@pytest.fixture
def simulate_pois_dataloader():
  x, llik = _simulate_pois()
  n, p = x.shape
  s = x.sum(axis=1)
  b = 64
  data = anmf.dataset.ExprDataset(x, s)
  collate_fn = getattr(data, 'collate_fn', td.dataloader.default_collate)
  data = td.DataLoader(data, batch_size=b, collate_fn=collate_fn)
  return data, n, p, b, llik

def _simulate_gamma():
  n = 500
  p = 10
  np.random.seed(0)
  # Typical values (Sarkar et al. PLoS Genet 2019)
  log_mu = np.random.uniform(-12, -6, size=(1, p))
  log_phi = np.random.uniform(-6, 0, size=(1, p))
  s = np.random.poisson(lam=1e5, size=(n, 1))
  # Important: NB success probability is (n, p)
  F = st.nbinom(n=np.exp(-log_phi), p=1 / (1 + s.dot(np.exp(log_mu + log_phi))))
  x = F.rvs()
  llik = F.logpmf(x).sum()
  return x, s, log_mu, log_phi, llik

@pytest.fixture
def simulate_gamma():
  return _simulate_gamma()

@pytest.fixture
def simulate_point_gamma():
  x, s, log_mu, log_phi, _ = _simulate_gamma()
  n, p = x.shape
  logodds = np.random.uniform(-3, -1, size=(1, p))
  pi0 = sp.expit(logodds)
  z = np.random.uniform(size=x.shape) < pi0
  y = np.where(z, 0, x)
  F = st.nbinom(n=np.exp(-log_phi), p=1 / (1 + s.dot(np.exp(log_mu + log_phi))))
  llik_nonzero = np.log(1 - pi0) + F.logpmf(y)
  llik = np.where(y < 1, np.log(pi0 + np.exp(llik_nonzero)), llik_nonzero).sum()
  return y, s, log_mu, log_phi, logodds, llik
