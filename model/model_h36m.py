from torch.nn import Module
from torch import nn
import torch
import math
import utils.util as util
from siMLPe import mlp_4d
import numpy as np
from torch.nn.parameter import Parameter
from utils.opt import Options
import sys
from model import BaseModel as BaseBlock

class MultiStageModel_4(Module):
    def __init__(self, opt):
        super(MultiStageModel_4, self).__init__()

        self.opt = opt
        self.kernel_size = opt.kernel_size
        self.d_model = opt.d_model
        self.dct_n = opt.dct_n
        self.in_features = opt.in_features
        self.num_stage = opt.num_stage
        self.node_n = self.in_features // 3
        self.input_n = opt.input_n
        self.output_n = opt.output_n

        # 模型参数验证
        assert opt.kernel_size == 10, "Kernel size must be 10"
        
        # # 权重参数
        self.w1 = nn.Parameter(torch.tensor(0.07, requires_grad=True))
        self.w2 = nn.Parameter(torch.tensor(0.28, requires_grad=True))
        self.w3 = nn.Parameter(torch.tensor(0.65, requires_grad=True))
        self.w4 = nn.Parameter(torch.tensor(0.1, requires_grad=True))  # 特殊分支权重s

        # 编码器-解码器结构
        self.encoder_layer_num = 1
        self.decoder_layer_num = 1
        
        # 卷积层用于特征融合
        self.conv70_35 = torch.nn.Conv2d(in_channels=70, out_channels=35, kernel_size=(1,1))
        
        self.final_proj1 = nn.Conv2d(3, self.d_model, kernel_size=1)
        self.final_proj2 = nn.Conv2d(3, self.d_model, kernel_size=1)
        self.final_proj3 = nn.Conv2d(3, self.d_model, kernel_size=1)
        self.final_proj4 = nn.Conv2d(3, self.d_model, kernel_size=1)
        
        # 四个解码器分支
        dec_config = {
            'in_channal': self.d_model,
            'out_channal': self.d_model,
            'n_txcnn_layers': opt.n_tcnn_layers,
            'txc_kernel_size': opt.tcnn_kernel_size,
            'txc_dropout': opt.tcnn_dropout,
            'node_n': self.node_n,
            'seq_len': self.dct_n,
            'p_dropout': opt.drop_out,
            'num_stage': self.decoder_layer_num,
            'snum_stage': 2
        }
        self.gcn_decoder1 = BaseBlock.GCN_decoder(**dec_config)
        self.gcn_decoder2 = BaseBlock.GCN_decoder(**dec_config)
        self.gcn_decoder3 = BaseBlock.GCN_decoder(**dec_config)
        self.gcn_decoder4 = BaseBlock.GCN_decoder(**dec_config)

        # 窗口大小参数
        self.window_size = 5
        
        # 动态确定需要更新的帧数
        self.num_update_frames = self.window_size - 1 if self.window_size > 1 else 1
        
        # 动态创建GCN残差模块
        self.gcn_blocks = nn.ModuleList([
            BaseBlock.GraphConvolution(
                in_c=3,
                out_c=3,
                node_n=self.node_n,
                seq_len=self.window_size - 1,
                bias=True
            ) for _ in range(self.num_update_frames)
        ])
        
        # MLP处理模块
        self.mlp_blockt = mlp_4d.MLPblock(
            dim=self.node_n, seq=self.window_size,
            use_norm=True, 
            use_spatial_fc=False,
            layernorm_axis='spatial'
        )
        self.mlp_blocks = mlp_4d.MLPblock(
            dim=self.node_n, seq=self.window_size,
            use_norm=True, 
            use_spatial_fc=True,
            layernorm_axis='temporal'
        )

    def forward(self, src):
        bs = src.shape[0]
        dct_n = src.shape[1]
        gt = src.clone()

        # 生成输入序列 [b,T,66]
        idx = list(range(self.kernel_size)) + [self.kernel_size - 1] * (dct_n - self.kernel_size)
        input_gcn1 = src[:, idx].clone()

        # 计算归一化权重
        total_weight = self.w1 + self.w2 + self.w3 + self.w4
        w1_normalized = self.w1 / total_weight
        w2_normalized = self.w2 / total_weight
        w3_normalized = self.w3 / total_weight
        w4_normalized = self.w4 / total_weight

        # DCT变换准备
        dct_m, idct_m = util.get_dct_matrix(dct_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.opt.cuda_idx)
        idct_m = torch.from_numpy(idct_m).float().to(self.opt.cuda_idx)

        # 输入预处理 [b,66,dct_n]->[b,3,22,dct_n]
        input_gcn_dct = torch.einsum('ij,bjk->bik', dct_m[:dct_n], input_gcn1)
        input_gcn_dct = input_gcn_dct.permute(0, 2, 1)
        input_gcn_dct = input_gcn_dct.reshape(bs, self.node_n, -1, dct_n).permute(0, 2, 1, 3) # [bs, 3, self.node_n, dct_n]

        # ================= 主处理分支 =================
        # 第一级
        latent_gcn_dct = self.final_proj1(input_gcn_dct) # [bs, 16, self.node_n, dct_n]
        output_dct_1 = self.gcn_decoder1(latent_gcn_dct)[:, :, :, :dct_n]

        # 第二级
        latent_gcn_dct = self.final_proj2(output_dct_1) # [bs, 16, self.node_n, dct_n]
        output_dct_2 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        # 第三级
        latent_gcn_dct = self.final_proj2(output_dct_2) # [bs, 16, self.node_n, dct_n]
        output_dct_3 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        # 第四级
        latent_gcn_dct = self.final_proj2(output_dct_3) # [bs, 16, self.node_n, dct_n]
        output_dct_4 = self.gcn_decoder2(latent_gcn_dct)[:, :, :, :dct_n]

        # # 加权融合三路输出
        com_out_fin = output_dct_1 * w1_normalized + output_dct_2 * w2_normalized + output_dct_3 * w3_normalized + output_dct_4 * w4_normalized # [bs, 3, self.node_n, dct_n]

        # ================= 精确残差逐帧更新 =================
        output_updated = {}  # 存储更新后的帧
        output_list = []     # 最终输出序列
        
        # 第一步：确定需要更新的帧范围
        # 对于N=2: 更新帧10 (索引10)
        # 对于N=3: 更新帧9-10 (索引9-10)
        # 对于N=4: 更新帧8-10 (索引8-10)
        # 对于N=5: 更新帧7-10 (索引7-10)
        update_start = self.input_n - (self.window_size - 1) + 1
        update_end = self.input_n  # 10
        update_indices = list(range(update_start, update_end + 1))
        
        # 第二步：按顺序更新这些帧（使用原始数据计算残差）
        for idx, update_idx in enumerate(update_indices):
            # 确定上下文窗口（前window_size-1帧）
            context_start = update_idx - (self.window_size - 1)
            context_frames = list(range(context_start, update_idx))
            
            # 构建输入张量 [原始上下文帧 + 当前帧]
            input_frames = com_out_fin[:, :, :, context_start: update_idx + 1] # [b, c, n, window_size]
            
            # 计算残差（仅对输入范围内的帧）
            residual_input = com_out_fin[:, :, :, context_start: update_idx] - \
                            gt.reshape(bs, -1, self.node_n, dct_n)[:, :, :, context_start: update_idx]
            
            output_of_differ = self.gcn_blocks[idx](residual_input)
 
            # 双MLP处理
            midoutput = self.mlp_blocks(input_frames)
            
            # 残差注入
            midoutput[:, :, :, :self.window_size - 1] += output_of_differ
            
            output_frame = self.mlp_blockt(midoutput)
            
            # 只保留当前帧的更新结果
            updated_frame = output_frame[:, :, :, -1]
            output_updated[update_idx] = updated_frame
        output_list.append(output_updated[self.input_n].unsqueeze(-1))  # 添加最后更新的帧
        
        # 第三步：预测后续帧（使用更新后的帧）
        predict_start = self.input_n + 1  # 11
        predict_end = self.input_n + self.output_n - 1 # 34
        
        for predict_idx in range(predict_start, predict_end + 1):
            # 构建输入窗口 [更新后的前window_size帧]
            context_start = predict_idx - self.window_size + 1
            input_frames = []
            
            # 添加上下文帧
            for frame_idx in range(context_start, predict_idx):
                input_frames.append(output_updated[frame_idx].unsqueeze(-1))
            input_frames.append(com_out_fin[:, :, :, predict_idx].unsqueeze(-1))  # 添加当前帧
            
            input_tensor = torch.cat(input_frames, dim=-1)  # [b, c, n, window_size]
            
            # 双MLP处理
            midoutput = self.mlp_blocks(input_tensor)
            output_frame = self.mlp_blockt(midoutput)
            
            # 取最后一帧作为预测结果
            predicted_frame = output_frame[:, :, :, -1]
            output_updated[predict_idx] = predicted_frame
            output_list.append(predicted_frame.unsqueeze(-1))
        
        output_concatenated = torch.cat(output_list, dim=-1)  # [b, c, n, output_n]
        # 重构最终输出 [原始输入帧0-9 + 更新/预测帧]
        output_dct_fin = torch.cat((
            com_out_fin[:, :, :, :self.input_n],  # 原始输入帧0-9
            output_concatenated # 更新/预测帧
        ), dim=3)

        # ================= 输出处理 =================
        # 主分支输出转换
        output_dct_1 = output_dct_1.permute(0, 2, 1, 3).reshape(bs, -1, dct_n) # [bs, 66, dct_n]
        output_dct_2 = output_dct_2.permute(0, 2, 1, 3).reshape(bs, -1, dct_n) # [bs, 66, dct_n]
        output_dct_3 = output_dct_3.permute(0, 2, 1, 3).reshape(bs, -1, dct_n) # [bs, 66, dct_n]
        output_dct_4 = output_dct_4.permute(0, 2, 1, 3).reshape(bs, -1, dct_n) # [bs, 66, dct_n]
        output_dct_fin = output_dct_fin.permute(0, 2, 1, 3).reshape(bs, -1, dct_n) # [bs, 66, dct_n]
        
        output_1 = torch.matmul(idct_m[:, :dct_n], output_dct_1.permute(0, 2, 1)) # [bs, dct, 66]
        output_2 = torch.matmul(idct_m[:, :dct_n], output_dct_2.permute(0, 2, 1)) # [bs, dct, 66]
        output_3 = torch.matmul(idct_m[:, :dct_n], output_dct_3.permute(0, 2, 1)) # [bs, dct, 66]
        output_4 = torch.matmul(idct_m[:, :dct_n], output_dct_4.permute(0, 2, 1)) # [bs, dct, 66]
        output_fin = torch.matmul(idct_m[:, :dct_n], output_dct_fin.permute(0, 2, 1)) # [bs, dct, 66]
        
        # 分支融合
        output_1 = torch.cat((output_1, input_gcn1), dim=1)
        output_2 = torch.cat((output_2, input_gcn1), dim=1)
        output_3 = torch.cat((output_3, input_gcn1), dim=1)
        output_4 = torch.cat((output_4, input_gcn1), dim=1)
        output_fin = torch.cat((output_fin, input_gcn1), dim=1)
        
        # 降维处理
        output_1 = self.conv70_35(output_fin.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_2 = self.conv70_35(output_fin.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_3 = self.conv70_35(output_fin.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_4 = self.conv70_35(output_fin.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)
        output_fin = self.conv70_35(output_fin.reshape(bs, dct_n * 2, self.node_n, -1)).reshape(bs, dct_n, -1)

        return output_fin, output_4, output_3, output_2, output_1

if __name__ == '__main__':
    option = Options().parse()
    option.d_model = 64
    model = MultiStageModel_4(opt=option).cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    src = torch.randn(32, 35, 66).cuda()  # [bs, seq_len, features]
    outputs = model(src)