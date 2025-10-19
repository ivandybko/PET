import os
import argparse
import numpy as np
import torch
import torch.distributed as dist
from .utils import not_share, and_beaver

def reveal(d_local, e_local):
    # Преобразование данных в тензоры 
    if isinstance(d_local, np.ndarray):
        d_t = torch.from_numpy(d_local.astype(np.uint8)).to(dtype=torch.uint8)
        e_t = torch.from_numpy(e_local.astype(np.uint8)).to(dtype=torch.uint8)
    else:
        d_t = torch.tensor([int(d_local)], dtype=torch.uint8)
        e_t = torch.tensor([int(e_local)], dtype=torch.uint8)

    # Обмен долями  
    d_buf = [torch.zeros_like(d_t) for _ in range(dist.get_world_size())]
    e_buf = [torch.zeros_like(e_t) for _ in range(dist.get_world_size())]
    dist.all_gather(d_buf, d_t)
    dist.all_gather(e_buf, e_t)

    # Восстановление значений d и e
    d_open = d_buf[0].clone()
    e_open = e_buf[0].clone()
    for i in range(1, len(d_buf)):
        d_open = d_open ^ d_buf[i]
        e_open = e_open ^ e_buf[i]

    if isinstance(d_local, np.ndarray):
        return d_open.cpu().numpy().astype(np.uint8), e_open.cpu().numpy().astype(np.uint8)
    else:
        return int(d_open.item()), int(e_open.item())

def load_data(rank: int):
    data_dir = f"./data/party{rank}"
    a_bits = np.load(os.path.join(data_dir, "a_bits.npy"))
    b_bits = np.load(os.path.join(data_dir, "b_bits.npy"))
    triples = np.load(os.path.join(data_dir, "beaver_triples.npy"))
    return a_bits.astype(np.uint8), b_bits.astype(np.uint8), triples.astype(np.uint8)

def bitwise_comparator(a_bits_share, b_bits_share, triples, open_and_return, rank):
    nbits = len(a_bits_share)
    triple_idx = 0
    borrow_share = 0

    for i in range(nbits - 1, -1, -1):
        ai = int(a_bits_share[i])
        bi = int(b_bits_share[i])

        not_ai = not_share(ai) if rank == 0 else ai

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term1 = and_beaver(not_ai, bi, a_s, b_s, c_s, open_and_return, rank)

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term2 = and_beaver(not_ai, borrow_share, a_s, b_s, c_s, open_and_return, rank)

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term3 = and_beaver(bi, borrow_share, a_s, b_s, c_s, open_and_return, rank)

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term12 = and_beaver(term1, term2, a_s, b_s, c_s, open_and_return, rank)

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term13 = and_beaver(term1, term3, a_s, b_s, c_s, open_and_return, rank)

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term23 = and_beaver(term2, term3, a_s, b_s, c_s, open_and_return, rank)

        a_s = int(triples[triple_idx, 0])
        b_s = int(triples[triple_idx, 1])
        c_s = int(triples[triple_idx, 2])
        triple_idx += 1
        term123 = and_beaver(term12, term3, a_s, b_s, c_s, open_and_return, rank)

        # x ∨ y ∨ z = x ⊕ y ⊕ z ⊕ (x∧y) ⊕ (x∧z) ⊕ (y∧z) ⊕ (x∧y∧z)
        borrow_share = term1 ^ term2 ^ term3 ^ term12 ^ term13 ^ term23 ^ term123

    return int(borrow_share)


def reconstruct_and_print(local_share, rank):
    t = torch.tensor([int(local_share)], dtype=torch.uint8)
    buf = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(buf, t)
    rec = buf[0]
    for i in range(1, len(buf)):
        rec = rec ^ buf[i]
    rec_np = rec.cpu().numpy().astype(np.uint8)
    print(f"[rank {rank}] Reconstructed result (a<b) = {int(rec_np[0])}")

def main(rank: int, world_size: int):
    dist.init_process_group(backend="gloo")
    a_bits, b_bits, triples = load_data(rank)
    res_share = bitwise_comparator(a_bits, b_bits, triples, reveal, rank)
    reconstruct_and_print(res_share, rank)

