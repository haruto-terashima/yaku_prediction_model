import torch

def plus_set(all_img_set, all_multi_hot_labels):
    data_set = []
    for i, k in zip(all_img_set, all_multi_hot_labels):
        i = torch.tensor(i, dtype=torch.float32).unsqueeze(1)
        k = torch.tensor(k, dtype=torch.float32)
        data_set.append((i,k))
    return data_set