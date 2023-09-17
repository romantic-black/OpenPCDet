import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)  # 2
        # 卷积核大小为1的作用: 减少通道数，类似于全连接层
        # 相当于对输入通道进行线性变换
        # 空间感受野不存在，只针对某个点
        self.conv_cls = nn.Conv2d(  # Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(  # Conv2d(256, 14, kernel_size=(1, 1), stride=(1, 1))
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )
        # True
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(  # Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1))
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,    # 4
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        # 这种处理方式用于初始化不平衡类别的分类卷积层，使模型倾向于背景类别
        # -np.log((1 - pi) / pi) 其中 pi 需接近 0
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # 这里 std=0.001，说明 bbox 预测需接近原点（似乎就是锚框）
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        # pytorch 默认将线性层和卷积层权重进行 kaiming 初始化，对偏置初始化为0
        # kaiming 初始化是专为 relu 激活函数设计的初始化方式
        # 用于缓解 relu 函数导致的梯度消失或爆炸问题，尤其是在非常深的网络中
        # 具体流程：权重用均值为0，方差为2/n进行初始化
        # 对线性层，n为输入神经元数量
        # 对卷积层，n为输入通道数*卷积核大小
        # 初始化的目的是保持每一层输出的方差与输入的方差相近，
        # 从而避免深度网络中方差的逐层放大，缓解梯度消失或爆炸的问题。

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        # 这时候和锚框没有任何关系，就是 bev 特征图大小
        cls_preds = self.conv_cls(spatial_features_2d)  # [4, 2, 200, 176]
        box_preds = self.conv_box(spatial_features_2d)  # [4, 14, 200, 176]

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:   # True
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:   # True
            targets_dict = self.assign_targets(     # 获取锚框与 gt 的关系
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)      # 这里没用上，应该是基类使用了

        if not self.training or self.predict_boxes_when_training:   # True
            # 输入每个锚框的预测结果（相对变换）获取 bbox
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            # batch_cls_preds: [B, 70400, 1]
            # batch_box_preds: [B, 70400, 7]
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
