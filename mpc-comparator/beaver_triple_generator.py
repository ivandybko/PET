import numpy as np
import os

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKING_DIR, "data")
PARTIES = ["party0", "party1"]
BITS = 16   
NUM_TRIPLES = 10000

# возвращает для каждой party их битовые доли (party0_bits, party1_bits)
def share_bitwise_number(x, bits=BITS):     
    bits_list = [(x >> i) & 1 for i in reversed(range(bits))] # получение битового представления 
    p0 = []
    p1 = []
    for b in bits_list:
        r = np.random.randint(0,2)
        p0.append(r)
        p1.append(b ^ r)
    return p0, p1

def gen_beaver_triples(num, bits=BITS):
    a = np.random.randint(0,2)
    b = np.random.randint(0,2)
    c = a & b
    a0 = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    a1 = a ^ a0
    b0 = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    b1 = b ^ b0
    c0 = np.random.randint(0,2,size=(num,), dtype=np.uint8)
    c1 = c ^ c0
    triples_party0 = np.stack([a0, b0, c0], axis=1)
    triples_party1 = np.stack([a1, b1, c1], axis=1)
    return triples_party0, triples_party1

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    a_val = 1
    b_val = 10
    bits = BITS

    # Разделение секрета
    a0_bits, a1_bits = share_bitwise_number(a_val, bits=bits)
    b0_bits, b1_bits = share_bitwise_number(b_val, bits=bits)

    # Генерация троек Бивера 
    t0, t1 = gen_beaver_triples(NUM_TRIPLES, bits=bits)

    for i, name in enumerate(PARTIES):
        os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)
        if i == 0:
            np.save(os.path.join(DATA_DIR,name,"a_bits.npy"), a0_bits)
            np.save(os.path.join(DATA_DIR,name,"b_bits.npy"), b0_bits)
            np.save(os.path.join(DATA_DIR,name,"beaver_triples.npy"), t0)
        else:
            np.save(os.path.join(DATA_DIR,name,"a_bits.npy"), a1_bits)
            np.save(os.path.join(DATA_DIR,name,"b_bits.npy"), b1_bits)
            np.save(os.path.join(DATA_DIR,name,"beaver_triples.npy"), t1)

    print("Generated data in ./data/")


if __name__ == "__main__":
    main()
