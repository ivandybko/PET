import numpy as np
import torch
import struct

# def xor_share(x_share, y_share):
#     return x_share ^ y_share

def not_share(x_share):
    return 1 ^ x_share

# AND using Beaver triple (binary)
# def and_beaver(x_share, y_share, a_share, b_share, c_share, reveal_and_return):
#     d_local = x_share ^ a_share
#     e_local = y_share ^ b_share
#     d_open, e_open = reveal_and_return(d_local, e_local)  # both ints or arrays
#     # d_open, e_open are plain ints (0/1) (or arrays if vectorized)
#     # z_share = c_share ^ (d_open & b_share) ^ (e_open & a_share) ^ (d_open & e_open)
#     # note: (d_open & b_share) is local: if b_share is array, elementwise
#     z_share = c_share ^ (d_open & b_share) ^ (e_open & a_share) ^ (d_open & e_open)
#     return z_share

def and_beaver(x_share, y_share, a_share, b_share, c_share, reveal_func, rank):

    d_share = x_share ^ a_share
    e_share = y_share ^ b_share
    
    # Открываем d и e
    d_open, e_open = reveal_func(d_share, e_share)
    
    # Вычисляем d ∧ b_share
    if isinstance(d_open, np.ndarray):
        d_and_b = np.bitwise_and(d_open, b_share)
        e_and_a = np.bitwise_and(e_open, a_share)
        d_and_e = np.bitwise_and(d_open, e_open)
    else:
        d_and_b = int(d_open) & int(b_share)
        e_and_a = int(e_open) & int(a_share)
        d_and_e = int(d_open) & int(e_open)
    
    # Только party 0 добавляет d ∧ e 
    if rank == 0:
        z_share = c_share ^ d_and_b ^ e_and_a ^ d_and_e
    else:
        z_share = c_share ^ d_and_b ^ e_and_a
    
    return z_share
