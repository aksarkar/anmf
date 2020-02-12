import anmf.modules
import numpy as np
import torch

def test_Encoder():
  enc = anmf.modules.Encoder(input_dim=100, hidden_dim=50, output_dim=10)
  x = torch.zeros([1, 100])
  l = enc.forward(x).detach()
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

def test_Pois_forward():
  dec = anmf.modules.Pois(input_dim=10, output_dim=100)
  torch.manual_seed(0)
  l = torch.nn.functional.softmax(torch.randn([1, 10]), dim=1)
  lam = dec.forward(l).detach()
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
