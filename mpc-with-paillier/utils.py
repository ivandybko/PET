import torch
import pickle

def send(obj, dst_rank):
    import pickle
    import torch

    # сериализация Python-объекта в байты
    data = pickle.dumps(obj)

    # создаём тензор длины данных (1 элемент)
    length_tensor = torch.tensor([len(data)], dtype=torch.long)

    # отправляем длину
    torch.distributed.send(length_tensor, dst=dst_rank)

    # отправляем сами байты
    byte_tensor = torch.tensor(list(data), dtype=torch.uint8)
    torch.distributed.send(byte_tensor, dst=dst_rank)


def recv(src_rank):
    length = torch.zeros(1, dtype=torch.long)
    torch.distributed.recv(length, src=src_rank)
    n = int(length.item())
    byte_tensor = torch.empty(n, dtype=torch.uint8)
    torch.distributed.recv(byte_tensor, src=src_rank)
    data = bytes(byte_tensor.tolist())
    obj = pickle.loads(data)
    return obj
