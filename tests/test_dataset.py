import anmf
import scipy.sparse as ss
import torch
import torch.utils.data as td

from fixtures import *

def test_ExprDataset_init(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  y = ss.csr_matrix(x)
  data = anmf.dataset.ExprDataset(y, s)
  assert len(data) == y.shape[0]
  if torch.cuda.is_available():
    assert (data.data.cpu().numpy() == y.data).all()
    assert (data.indices.cpu().numpy() == y.indices).all()
    assert (data.indptr.cpu().numpy() == y.indptr).all()
  else:
    assert (data.data.numpy() == y.data).all()
    assert (data.indices.numpy() == y.indices).all()
    assert (data.indptr.numpy() == y.indptr).all()

def test_ExprDataset_init_dense(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = anmf.dataset.ExprDataset(x, s)
  if torch.cuda.is_available():
    y = ss.csr_matrix((data.data.cpu().numpy(), data.indices.cpu().numpy(), data.indptr.cpu().numpy()))
  else:
    y = ss.csr_matrix((data.data.numpy(), data.indices.numpy(), data.indptr.numpy()))
  assert (y.todense() == x).all()

def test_ExprDataset_init_coo(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  y = ss.coo_matrix(x)
  data = anmf.dataset.ExprDataset(y, s)
  if torch.cuda.is_available():
    z = ss.csr_matrix((data.data.cpu().numpy(), data.indices.cpu().numpy(), data.indptr.cpu().numpy())).tocoo()
  else:
    z = ss.csr_matrix((data.data.numpy(), data.indices.numpy(), data.indptr.numpy())).tocoo()
  # This is more efficient than ==
  assert not (y != z).todense().any()

def test_ExprDataset__get_item__(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = anmf.dataset.ExprDataset(x, s)
  y = data[0]
  assert y == 0

def test_ExprDataset_collate_fn(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = anmf.dataset.ExprDataset(x, s)
  batch_size = 10
  y, t = data.collate_fn(range(batch_size))
  if torch.cuda.is_available():
    assert (y.cpu().numpy() == x[:batch_size]).all()
    assert (t.cpu().numpy() == s[:batch_size]).all()
  else:
    assert (y.numpy() == x[:batch_size]).all()
    assert (t.numpy() == s[:batch_size]).all()

def test_ExprDataset_collate_fn_shuffle(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  data = anmf.dataset.ExprDataset(x, s)
  idx = [10, 20, 30, 40, 50]
  y, t = data.collate_fn(idx)
  if torch.cuda.is_available():
    assert (y.cpu().numpy() == x[idx]).all()
    assert (t.cpu().numpy() == s[idx]).all()
  else:
    assert (y.numpy() == x[idx]).all()
    assert (t.numpy() == s[idx]).all()

def test_ExprDataset_DataLoader(simulate_point_gamma):
  x, s, log_mu, log_phi, logodds, l0 = simulate_point_gamma
  batch_size = 10
  sparse_data = anmf.dataset.ExprDataset(x, s)
  data = td.DataLoader(sparse_data, batch_size=batch_size, shuffle=False, collate_fn=sparse_data.collate_fn)
  y, t = next(iter(data))
  if torch.cuda.is_available():
    assert (y.cpu().numpy() == x[:batch_size]).all()
    assert (t.cpu().numpy() == s[:batch_size]).all()
  else:
    assert (y.numpy() == x[:batch_size]).all()
    assert (t.numpy() == s[:batch_size]).all()
