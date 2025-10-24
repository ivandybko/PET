import torch, torch.distributed as dist
from phe import paillier
from utils import send, recv
import csv, os
from config import PAILLIER_KEY_SIZE, MPC_MODULO

OUT_DIR = os.environ.get('OUT_DIR', './shared')

def main(rank: int, world_size: int):
    dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=rank)
    N = 100  # Генерируем 100 троек 
    triples = []

    # Генерация и обмен публичным ключом один раз перед циклом
    if rank == 1:
        public_key, private_key = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_SIZE)
        send(public_key, dst_rank=0)
    elif rank == 0:
        pubkey = recv(src_rank=1)

    for _ in range(N):
        if rank == 0:
            a1 = int.from_bytes(os.urandom(16), 'big') % MPC_MODULO
            b1 = int.from_bytes(os.urandom(16), 'big') % MPC_MODULO

            # Получение зашифрованных a2, b2 (публичный ключ уже получен)
            enc_a2 = recv(src_rank=1)
            enc_b2 = recv(src_rank=1)
            r = int.from_bytes(os.urandom(16), 'big') % MPC_MODULO

            enc_a2_b1 = enc_a2 * b1
            enc_b2_a1 = enc_b2 * a1 
            enc_S = enc_a2_b1 + enc_b2_a1
            enc_r = pubkey.encrypt(r)
            enc_S_plus_r = enc_S + enc_r

            send(enc_S_plus_r, dst_rank=1)

            c1 = (a1 * b1 - r) % MPC_MODULO
            triples.append((a1, b1, c1))
        elif rank == 1:
            a2 = int.from_bytes(os.urandom(16), 'big') % MPC_MODULO
            b2 = int.from_bytes(os.urandom(16), 'big') % MPC_MODULO

            # Отправка зашифрованных значений a2, b2 Party 1 
            send(public_key.encrypt(a2), dst_rank=0)
            send(public_key.encrypt(b2), dst_rank=0)

            enc_S_plus_r = recv(src_rank=0)
            S_plus_r = private_key.decrypt(enc_S_plus_r)

            c2 = (a2 * b2 + S_plus_r) % MPC_MODULO
            triples.append((a2, b2, c2))

    # Синхронизация перед записью
    dist.barrier()

    if rank == 0:
        out_path = os.path.join(OUT_DIR, 'p1.csv')
    elif rank == 1:
        out_path = os.path.join(OUT_DIR, 'p2.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['a_share', 'b_share', 'c_share'])
        for a, b, c in triples:
            writer.writerow([a, b, c])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
