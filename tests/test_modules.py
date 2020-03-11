import anmf
import numpy as np
import scipy.stats as st
import torch
import torch.utils.data as td

from fixtures import *

def test_Encoder():
  enc = anmf.modules.Encoder(input_dim=100, hidden_dim=50, output_dim=10)
  x = torch.zeros([1, 100])
  # if torch.cuda.is_available():
  #   x = x.cuda()
  l = enc.forward(x)
  assert l.requires_grad
  l = l.detach()
  # if torch.cuda.is_available():
  #   l = l.cpu()
  l = l.numpy()
  assert l.shape == (1, 10)
  assert (l >= 0).all()

@require_cuda
def test_Encoder_cuda():
  enc = anmf.modules.Encoder(input_dim=100, hidden_dim=50, output_dim=10).cuda()
  x = torch.zeros([1, 100]).cuda()
  l = enc.forward(x)
  assert l.requires_grad
  l = l.detach()
  l = l.cpu().numpy()
  assert l.shape == (1, 10)
  assert (l >= 0).all()

def test_Encoder_minibatch():
  enc = anmf.modules.Encoder(input_dim=100, hidden_dim=50, output_dim=10)
  x = torch.randn([10, 100])
  l = enc.forward(x).detach()
  l = l.numpy()
  assert l.shape == (10, 10)
  assert (l >= 0).all()
  
def test_Pois_parameters():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  assert len(list(dec.parameters())) == 1
  assert dec._f.requires_grad

def test_Pois_forward():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  torch.manual_seed(0)
  l = torch.nn.functional.softmax(torch.randn([1, 10]), dim=1)
  lam = dec.forward(l)
  assert lam.requires_grad
  lam = lam.detach().numpy()
  assert lam.shape == (1, 100)
  assert (lam >= 0).all()

@require_cuda
def test_Pois_forward_cuda():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100).cuda()
  torch.manual_seed(0)
  l = torch.nn.functional.softmax(torch.randn([1, 10]), dim=1).cuda()
  lam = dec.forward(l)
  assert lam.requires_grad
  lam = lam.cpu().detach().numpy()
  assert lam.shape == (1, 100)
  assert (lam >= 0).all()

def test_Pois_forward_minibatch():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  torch.manual_seed(0)
  l = torch.nn.functional.softmax(torch.randn([10, 10]), dim=1)
  lam = dec.forward(l).detach()
  lam = lam.numpy()
  assert lam.shape == (10, 100)
  assert (lam >= 0).all()

def test_ANMF():
  m = anmf.modules.ANMF(input_dim=100, latent_dim=3)

def test_ANMF_loss(simulate_pois_dataloader):
  data, n, p, b, _ = simulate_pois_dataloader
  m = anmf.modules.ANMF(input_dim=p, latent_dim=3)
  # Important: data loader moves to the GPU if available
  if torch.cuda.is_available:
    m.cuda()
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
  if torch.cuda.is_available:
    m.cuda()
  batch, size = next(iter(data))
  lam = m.denoise(batch)
  assert lam.shape == (b, p)
  assert (lam >= 0).all()

@require_cuda
def test_ANMF_oracle(simulate_pois_dataloader):
  data, n, p, b, oracle_llik = simulate_pois_dataloader
  m = anmf.modules.ANMF(input_dim=p, latent_dim=3)
  m.cuda()
  m.fit(data, max_epochs=50)
  llik = np.array([st.poisson(mu=size.cpu().numpy().reshape(-1, 1) * m.denoise(batch)).logpmf(batch.cpu().numpy()).sum()
                   for batch, size in data]).sum()
  assert llik <= oracle_llik
