import numpy as np
import os

BITS = 16
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PARTIES = ["party0", "party1"]

# Загрузка данных
a_bits_party0 = np.load(os.path.join(DATA_DIR, "party0", "a_bits.npy"))
b_bits_party0 = np.load(os.path.join(DATA_DIR, "party0", "b_bits.npy"))
a_bits_party1 = np.load(os.path.join(DATA_DIR, "party1", "a_bits.npy"))
b_bits_party1 = np.load(os.path.join(DATA_DIR, "party1", "b_bits.npy"))
triples_party0 = np.load(os.path.join(DATA_DIR, "party0", "beaver_triples.npy"))
triples_party1 = np.load(os.path.join(DATA_DIR, "party1", "beaver_triples.npy"))

# Реконструкция битов
a_bits_reconstructed = a_bits_party0 ^ a_bits_party1
b_bits_reconstructed = b_bits_party0 ^ b_bits_party1

print("Reconstructed a_bits =", a_bits_reconstructed)
print("Reconstructed b_bits =", b_bits_reconstructed)

def bits_to_int(bits):
    return sum(bit * (2 ** (BITS - 1 - i)) for i, bit in enumerate(bits)) 

a_value = bits_to_int(a_bits_reconstructed)
b_value = bits_to_int(b_bits_reconstructed)
print("Reconstructed a =", a_value)
print("Reconstructed b =", b_value)


