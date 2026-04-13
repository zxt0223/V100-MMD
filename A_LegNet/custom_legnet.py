import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from mmcv.cnn import build_norm_layer
from mmdet.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class LWEGNet(BaseModule):
    def __init__(self, stem_dim=32, depths=(1, 4, 8, 2), fork_feat=True, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.fork_feat = fork_feat
        # Stem 层
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_dim, 7, stride=2, padding=3),
            build_norm_layer(dict(type='BN'), stem_dim)[1],
            nn.ReLU()
        )
        self.stages = nn.ModuleList()
        # 构建 4 个阶段，适配 Small 的 (1, 4, 8, 2) 深度
        for i in range(len(depths)):
            dim = int(stem_dim * 2**i)
            # 简化版 Block 堆叠
            stage = nn.Sequential(*[nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.Conv2d(dim, dim, 1),
                nn.ReLU()
            ) for _ in range(depths[i])])
            self.stages.append(stage)
            
            if i < len(depths) - 1:
                # 下采样层
                downsample = nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1),
                    build_norm_layer(dict(type='BN'), dim * 2)[1]
                )
                self.stages.append(downsample)

        self.out_indices = [0, 2, 4, 6]
        for i, idx in enumerate(self.out_indices):
            self.add_module(f'norm{idx}', build_norm_layer(dict(type='BN'), int(stem_dim * 2**i))[1])

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and i in self.out_indices:
                outs.append(getattr(self, f'norm{i}')(x))
        return tuple(outs)
