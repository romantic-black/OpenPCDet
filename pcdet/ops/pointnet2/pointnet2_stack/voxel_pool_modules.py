import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List


class NeighborVoxelSAModuleMSG(nn.Module):
                 
    def __init__(self, *, query_ranges: List[List[int]], radii: List[float], 
        nsamples: List[int], mlps: List[List[int]], use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)
        
        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()

        # query_ranges: [[4,4,4]], radii: [0.4], nsamples: [16], mlps:[[32/64, 32, 32]]
        for i in range(len(query_ranges)):  # len(query_ranges) = 1
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i]

            cur_mlp_in = nn.Sequential(
                nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[1])
            )
            
            cur_mlp_pos = nn.Sequential(
                nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp_spec[1])
            )

            cur_mlp_out = nn.Sequential(
                nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_spec[2]),
                nn.ReLU()
            )

            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                # 这段似乎是按照默认条件做的
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # 也是默认条件
                # BatchNorm 层公式为: $y=\gamma\left(\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\right)+\beta$
                # 其中 gamma 与 beta 是可以被学习的，且一般设为 1 和 0
                # mu 与 sigma 是 x 的均值和方差
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, \
                                        new_coords, features, voxel2point_indices):
        """
        M1+M2: 指多个 batch 的 RoI 网格点坐标被拼起来
        N1+N2: 指多个 batch 的稀疏有效点坐标被拼起来
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        # change the order to [batch_idx, z, y, x]
        # new_coords 是roi网格点经降采样后的坐标（因为特征和xyz都经过降采样，new_xyz没有）
        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        new_features_list = []
        for k in range(len(self.groupers)):     # range(0,1) , 遍历多组MLP
            # features_in: (1, C, M1+M2)
            features_in = features.permute(1, 0).unsqueeze(0)        # [1, 32, 110592]
            # conv1d 输入需要 [batch_size, num_channels, length]
            # 由于kernel=1，相当于通道内进行线性变换，然后batchnorm
            features_in = self.mlps_in[k](features_in)
            # features_in: (1, M1+M2, C)
            features_in = features_in.permute(0, 2, 1).contiguous()  # [1, 110592, 32]
            # features_in: (M1+M2, C)
            features_in = features_in.view(-1, features_in.shape[-1])   # [110592, 32]
            # grouped_features: (M1+M2, C, nsample) 相邻体素坐标
            # grouped_xyz: (M1+M2, 3, nsample)  相邻体素特征
            # VoxelQueryAndGrouping
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](
                new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices
            )
            grouped_features[empty_ball_mask] = 0

            # grouped_features: (1, C, M1+M2, nsample)
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            # grouped_xyz: (M1+M2, 3, nsample), 转为相对网格点中心的坐标
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            # grouped_xyz: (1, 3, M1+M2, nsample)
            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            # grouped_xyz: (1, C, M1+M2, nsample)
            # Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            position_features = self.mlps_pos[k](grouped_xyz)
            new_features = grouped_features + position_features
            new_features = self.relu(new_features)
            
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            # Conv1d(32, 32, kernel_size=(1,), stride=(1,), bias=False)
            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)
        
        # (M1 + M2 ..., C)  拼接多组MLP的结果
        new_features = torch.cat(new_features_list, dim=1)
        return new_features

