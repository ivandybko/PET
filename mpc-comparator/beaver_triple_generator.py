import numpy as np
import os
import shutil

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKING_DIR, "data")
PARTIES = ["party0", "party1"]
BITS = 16
NUM_TRIPLES = 10000

def share_bitwise_number(x, bits=BITS):
    bits_list = [(x >> i) & 1 for i in reversed(range(bits))]
    p0 = []
    p1 = []
    for b in bits_list:
        r = np.random.randint(0,2)
        p0.append(r)
        p1.append(b ^ r)
    return np.array(p0, dtype=np.uint8), np.array(p1, dtype=np.uint8)

def gen_beaver_triples(num):
    a = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    b = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    c = a & b
    a0 = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    b0 = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    c0 = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    a1 = a ^ a0
    b1 = b ^ b0
    c1 = c ^ c0
    triples_party0 = np.stack([a0, b0, c0], axis=1)
    triples_party1 = np.stack([a1, b1, c1], axis=1)
    return triples_party0, triples_party1

def main():
    # очистка старых данных
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

    a_val = 20
    b_val = 100

    a0_bits, a1_bits = share_bitwise_number(a_val)
    b0_bits, b1_bits = share_bitwise_number(b_val)
    t0, t1 = gen_beaver_triples(NUM_TRIPLES)

    for i, name in enumerate(PARTIES):
        path = os.path.join(DATA_DIR, name)
        os.makedirs(path)
        if i == 0:
            np.save(os.path.join(path,"a_bits.npy"), a0_bits)
            np.save(os.path.join(path,"b_bits.npy"), b0_bits)
            np.save(os.path.join(path,"beaver_triples.npy"), t0)
        else:
            np.save(os.path.join(path,"a_bits.npy"), a1_bits)
            np.save(os.path.join(path,"b_bits.npy"), b1_bits)
            np.save(os.path.join(path,"beaver_triples.npy"), t1)

    print("Generated data in ./data/")

if __name__ == "__main__":
    main()
