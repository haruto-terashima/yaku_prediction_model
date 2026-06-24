import torch
import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_models(num_labels):
    
    model = nn.Sequential(
        # 1〜3層目: 視野を広げる (BatchNormを追加)
        nn.Conv2d(4, 32, kernel_size=(1, 3), padding=(0, 1)),
        nn.BatchNorm2d(32), nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=(1, 5), padding=(0, 2)),
        nn.BatchNorm2d(64), nn.ReLU(),

        nn.Conv2d(64, 128, kernel_size=(1, 7), padding=(0, 3)),
        nn.BatchNorm2d(128), nn.ReLU(),

    # 4〜5層目: 高次の整理
        nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
        nn.BatchNorm2d(256), nn.ReLU(),

        nn.Conv2d(256, 256, kernel_size=(1, 3), padding=(0, 1)), # 512より256で安定
        nn.BatchNorm2d(256), nn.ReLU(),

    # 集約: 少しだけ情報を残す
        nn.AdaptiveAvgPool2d((1, 4)),
        nn.Flatten(),

    # 全結合層
        nn.Linear(256 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_labels)
    )

    model.apply(init_weights)
    return model
