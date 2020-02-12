import scipy.sparse as ss
import torch
import torch.utils.data as td

class ExprDataset(td.Dataset):
  """Specialized dataset for sparse count matrix and scale factors"""
  def __init__(self, x, s):
    super().__init__()
    if ss.issparse(x) and not ss.isspmatrix_csr(x):
      x = x.tocsr()
    else:
      x = ss.csr_matrix(x)
    self.n, self.p = x.shape
    if torch.cuda.is_available():
      device = 'cuda'
    else:
      device = 'cpu'
    self.data = torch.tensor(x.data, dtype=torch.float, device=device)
    self.indices = torch.tensor(x.indices, dtype=torch.long, device=device)
    self.indptr = torch.tensor(x.indptr, dtype=torch.long, device=device)
    self.s = torch.tensor(s, dtype=torch.float, device=device)

  def __getitem__(self, index):
    """Dummy implementation of __getitem__

    torch.utils.DataLoader.__next__() calls:

    batch = self.collate_fn([self.dataset[i] for i in indices])

    This is too slow, so instead of actually returning the data, like:

    start = self.indptr[index]
    end = self.indptr[index + 1]
    return (
      torch.sparse.FloatTensor(
        # Important: sparse indices are long in Torch
        torch.stack([torch.zeros(end - start, dtype=torch.long, device=self.indices.device), self.indices[start:end]]),
        # Important: this needs to be 1d before collate_fn
        self.data[start:end], size=[1, self.p]).to_dense().squeeze(),
      self.s[index]
    )

    and then concatenating in collate_fn, just return the index.
    """
    return index
    
  def __len__(self):
    return self.n

  def collate_fn(self, indices):
    """Return a minibatch of items

    Construct the entire sparse tensor in one shot, and then convert to
    dense. This is *much* faster than the default:

    torch.stack([data[i] for i in indices])

    """
    return (
      torch.sparse.FloatTensor(
        torch.cat([
          torch.stack([torch.full(((self.indptr[i + 1] - self.indptr[i]).item(),), j, dtype=torch.long, device=self.indices.device),
                       self.indices[self.indptr[i]:self.indptr[i + 1]]]) for j, i in enumerate(indices)], dim=1),
        torch.cat([self.data[self.indptr[i]:self.indptr[i + 1]] for i in indices]),
        size=[len(indices), self.p]).to_dense().squeeze(),
      self.s[indices]
    )
