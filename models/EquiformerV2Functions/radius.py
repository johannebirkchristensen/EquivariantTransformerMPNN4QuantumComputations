from typing import Optional
import torch
# be obs on this function... should be like this: https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/radius.py
# Wrapper: accepts batch_x / batch_y
def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
    ignore_same_index: bool = False
) -> torch.Tensor:
    # fast empty-check
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    # compute batch_size if not provided
    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None

    if batch_size > 1:
        if batch_x is None or batch_y is None:
            raise ValueError("batch_x and batch_y must be provided for batched inputs")
        # build pointers from batch vectors (assumes batch_x / batch_y are sorted)
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    # call the internal implementation that expects ptr_x, ptr_y BEFORE r
    return _radius_ptr(x, y, ptr_x, ptr_y, r,
                       max_num_neighbors, num_workers,
                       ignore_same_index)


def _radius_ptr(
    x: torch.Tensor,
    y: torch.Tensor,
    ptr_x: Optional[torch.Tensor],
    ptr_y: Optional[torch.Tensor],
    r: float,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    ignore_same_index: bool = False
) -> torch.Tensor:
    """Internal CPU fallback implementation.
    - ptr_x, ptr_y are optional tensors of length (batch_size+1) with boundaries,
      or None for single-batch.
    - For each y_j, finds x_i with ||x_i - y_j|| <= r.
    - Returns edge_index shaped (2, E) with rows=source indices (x), cols=target (y),
      matching the original torch-cluster behavior.
    """

    device = x.device
    r2 = float(r) * float(r)

    rows = []
    cols = []

    if ptr_x is None or ptr_y is None:
        # single-batch: brute-force pairwise distances
        # x: (N_x, F), y: (N_y, F)
        N_x = x.size(0)
        N_y = y.size(0)

        # compute squared distances: result shape (N_x, N_y)
        # do in blocks if huge? here we do full matrix (works for moderate sizes)
        diff = x.unsqueeze(1) - y.unsqueeze(0)               # (N_x, N_y, F)
        dist2 = diff.pow(2).sum(dim=2)                       # (N_x, N_y)

        for j in range(N_y):
            mask = dist2[:, j] <= r2                         # (N_x,)
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            if ignore_same_index:
                idx = idx[idx != j]
            if idx.numel() == 0:
                continue
            if idx.numel() > max_num_neighbors:
                # pick random subset
                perm = torch.randperm(idx.numel(), device=device)[:max_num_neighbors]
                idx = idx[perm]
            rows.append(idx)
            cols.append(torch.full((idx.numel(),), j, dtype=torch.long, device=device))

    else:
        # batched: use ptr boundaries
        batch_size = ptr_x.numel() - 1
        assert ptr_y.numel() == batch_size + 1
        for b in range(batch_size):
            x_s = int(ptr_x[b].item()); x_e = int(ptr_x[b + 1].item())
            y_s = int(ptr_y[b].item()); y_e = int(ptr_y[b + 1].item())

            if x_e - x_s == 0 or y_e - y_s == 0:
                continue

            xs = x[x_s:x_e]    # (nxb, F)
            ys = y[y_s:y_e]    # (nyb, F)

            # squared distances (nxb, nyb)
            diff = xs.unsqueeze(1) - ys.unsqueeze(0)
            dist2 = diff.pow(2).sum(dim=2)

            for jj in range(ys.size(0)):
                mask = dist2[:, jj] <= r2
                idx_local = torch.nonzero(mask, as_tuple=False).view(-1)
                if idx_local.numel() == 0:
                    continue
                if ignore_same_index:
                    # map local y index to global and compare if x/y are same set
                    global_y_index = y_s + jj
                    # compute global x indices and drop if equal to global_y_index
                    global_x_idx = idx_local + x_s
                    keep_mask = global_x_idx != global_y_index
                    idx_local = idx_local[keep_mask]
                    if idx_local.numel() == 0:
                        continue
                if idx_local.numel() > max_num_neighbors:
                    perm = torch.randperm(idx_local.numel(), device=device)[:max_num_neighbors]
                    idx_local = idx_local[perm]
                rows.append((idx_local + x_s))
                cols.append(torch.full((idx_local.numel(),), y_s + jj, dtype=torch.long, device=device))

    if len(rows) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)

    row = torch.cat(rows)
    col = torch.cat(cols)
    return torch.stack([row.long(), col.long()], dim=0)


def radius_graph(
    x: torch.Tensor,
    r: float,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = 'source_to_target',
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    assert flow in ['source_to_target', 'target_to_source']
    # the wrapper radius expects batch_x and batch_y; for graph we use x as both
    edge_index = radius(x, x, r, batch, batch,
                        max_num_neighbors,
                        num_workers, batch_size, not loop)
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]
    return torch.stack([row, col], dim=0)
