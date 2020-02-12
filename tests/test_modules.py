import anmf
import numpy as np
import scipy.stats as st
import torch
import torch.utils.data as td

from fixtures import *

def test_Encoder():
  enc = anmf.modules.Encoder(input_dim=100, hidden_dim=50, output_dim=10)
  x = torch.zeros([1, 100])
  l = enc.forward(x)
  assert l.requires_grad
  l = l.detach()
  if torch.cuda.is_available():
    l = l.cpu()
  l = l.numpy()
  assert l.shape == (1, 10)
  assert (l >= 0).all()
  assert (l <= 1).all()
  assert np.isclose(l.sum(), 1, rtol=0, atol=1e-6)

def test_Encoder_minibatch():
  enc = anmf.modules.Encoder(input_dim=100, hidden_dim=50, output_dim=10)
  x = torch.randn([10, 100])
  l = enc.forward(x).detach()
  if torch.cuda.is_available():
    l = l.cpu()
  l = l.numpy()
  assert l.shape == (10, 10)
  assert (l >= 0).all()
  assert (l <= 1).all()
  assert np.isclose(l.sum(axis=1), 1, rtol=0, atol=1e-6).all()
  
def test_Pois_parameters():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  assert len(list(dec.parameters())) == 1
  assert dec.logit_f.requires_grad

def test_Pois_forward():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  torch.manual_seed(0)
  l = torch.nn.functional.softmax(torch.randn([1, 10]), dim=1)
  lam = dec.forward(l)
  assert lam.requires_grad
  lam = lam.detach()
  if torch.cuda.is_available():
    lam = lam.cpu()
  lam = lam.numpy()
  assert lam.shape == (1, 100)
  assert (lam >= 0).all()
  assert (lam <= 1).all()
  assert np.isclose(lam.sum(axis=1), 1, rtol=0, atol=1e-6).all()

def test_Pois_forward_minibatch():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  torch.manual_seed(0)
  l = torch.nn.functional.softmax(torch.randn([10, 10]), dim=1)
  lam = dec.forward(l).detach()
  if torch.cuda.is_available():
    lam = lam.cpu()
  lam = lam.numpy()
  assert lam.shape == (10, 100)
  assert (lam >= 0).all()
  assert (lam <= 1).all()
  assert np.isclose(lam.sum(axis=1), 1, rtol=0, atol=1e-6).all()

def test_ANMF():
  m = anmf.modules.ANMF(input_dim=100, latent_dim=3)

def test_ANMF_loss(simulate_pois_dataloader):
  data, n, p, b, _ = simulate_pois_dataloader
  m = anmf.modules.ANMF(input_dim=p, latent_dim=3)
  batch, size = next(iter(data))
  l = m.loss(batch, size)
  assert l.requires_grad
  if torch.cuda.is_available():
    l = l.cpu()
  l = l.detach().numpy()
  assert l >= 0

def test_ANMF_denoise(simulate_pois_dataloader):
  data, n, p, b, _ = simulate_pois_dataloader
  m = anmf.modules.ANMF(input_dim=p, latent_dim=3)
  batch, size = next(iter(data))
  lam = m.denoise(batch)
  assert lam.shape == (b, p)
  assert (lam >= 0).all()
  assert (lam <= 1).all()

def test_ANMF_oracle(simulate_pois_dataloader):
  data, n, p, b, oracle_llik = simulate_pois_dataloader
  m = anmf.modules.ANMF(input_dim=p, latent_dim=3)
  m.fit(data, max_epochs=50)
  llik = np.array([st.poisson(mu=size.reshape(-1, 1) * m.denoise(batch)).logpmf(batch.numpy()).sum() for batch, size in data]).sum()
  assert llik <= oracle_llik
