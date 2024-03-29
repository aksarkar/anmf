"""Amortized inference for NMF"""
import torch

_softplus = torch.nn.functional.softplus

class GNMF(torch.nn.Module):
  """NMF by gradient descent"""
  def __init__(self, n, p, k):
    super().__init__()
    self._l = torch.nn.Parameter(torch.randn(size=[n, k]))
    self._f = torch.nn.Parameter(torch.randn(size=[p, k]))

  def loss(self, x):
    lam = torch.matmul(_softplus(self._l), _softplus(self._f).T)
    return -(x * torch.log(lam) - lam - torch.lgamma(x + 1)).sum()

  def fit(self, data, max_epochs=1000, trace=False, verbose=False, **kwargs):
    self.trace = []
    opt = torch.optim.RMSprop(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      for x in data:
        opt.zero_grad()
        loss = self.loss(x)
        if torch.isnan(loss):
          raise RuntimeError('nan loss')
        loss.backward()
        opt.step()
        if trace:
          self.trace.append(loss.detach().cpu().numpy())
      if verbose:
        print(f'[epoch={epoch}] loss={loss}')
    return self

  @property
  @torch.no_grad()
  def loadings(self):
    return _softplus(self._l).cpu().numpy()

  @property
  @torch.no_grad()
  def factors(self):
    return _softplus(self._f).cpu().numpy()

class Encoder(torch.nn.Module):
  """Encoder l_i = h(x_i)"""
  def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
    super().__init__()
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.BatchNorm1d(hidden_dim),
      torch.nn.Dropout(p=dropout),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.BatchNorm1d(hidden_dim),
      torch.nn.Dropout(p=dropout),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, output_dim),
      torch.nn.BatchNorm1d(output_dim),
      torch.nn.Softplus(),
    )

  def forward(self, x):
    return self.net(x)

class Pois(torch.nn.Module):
  """Decoder p(x_ij | l_ij, F) ~ Poisson(s_i x_ij F)"""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self._f = torch.nn.Parameter(torch.randn(size=[input_dim, output_dim]))

  def forward(self, x):
    """Return lambda_.j = (LF)_.j"""
    return torch.matmul(x, _softplus(self._f))

class ANMF(torch.nn.Module):
  """Amortized NMF"""
  def __init__(self, input_dim, hidden_dim=128, latent_dim=10):
    super().__init__()
    self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
    self.decoder = Pois(latent_dim, input_dim)

  def loss(self, x, s):
    """Return the negative log likelihood of x distributed as Poisson"""
    mean = self.decoder.forward(self.encoder.forward(x))
    s = s.reshape(-1, 1)
    return -(x * torch.log(s * mean) - s * mean - torch.lgamma(x + 1)).sum()

  def fit(self, data, max_epochs, trace=False, verbose=False, **kwargs):
    """Fit the model

    :param data: torch.Dataset
    :param kwargs: arguments to torch.optim.RMSprop

    """
    if trace:
      self.trace = []
    if torch.cuda.is_available():
      self.cuda()
    opt = torch.optim.RMSprop(self.parameters(), **kwargs)
    for epoch in range(max_epochs):
      for x, s in data:
        # We assume that x, s are already on the GPU if they can be
        opt.zero_grad()
        loss = self.loss(x, s)
        if torch.isnan(loss):
          raise RuntimeError('nan loss')
        loss.backward()
        opt.step()
        if trace:
          self.trace.append(loss.detach().cpu().numpy())
      if verbose:
        print(f'[epoch={epoch}] loss={loss}')
    return self

  @torch.no_grad()
  def denoise(self, x):
    if torch.cuda.is_available() and not x.is_cuda:
      x = x.cuda()
    lam = self.decoder.forward(self.encoder.forward(x))
    if torch.cuda.is_available():
      lam = lam.cpu()
    return lam.numpy()

  @torch.no_grad()
  def loadings(self, x):
    l = self.encoder.forward(x)
    if torch.cuda.is_available():
      l = l.cpu()
    return l.numpy()

  @torch.no_grad()
  def factors(self):
    f = _softplus(self.decoder._f)
    if torch.cuda.is_available():
      f = f.cpu()
    return f.numpy()
