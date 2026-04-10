import os
import sys
sys.path.append('../')
import time
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.view import view_gujia
from utils.geometric import GEOMETRIC
# ===== 数据与工具 =====
from utils import dpw3_3d as Datasets   # 3DPW 数据集
from utils import util, log, global_var
from utils.opt import Options
from model.model_h36m import MultiStageModel_4                # 模型定义（多阶段）
import torch.backends.cudnn as cudnn
import random
torch.backends.cudnn.enabled = False

# 设置环境变量解决CUDA错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ===== 关节与维度相关设置（3DPW） =====
# 说明：3DPW 的使用维度由数据集对象提供，这里占位初始化，
# 在加载数据集之后将以 dataset.dim_used 进行覆盖。
DIM_USED = None

def _resolve_ckpt_path(opt):
    ckpt = opt.ckpt
    if ckpt.endswith('.pth') or ckpt.endswith('.pth.tar'):
        return ckpt
    # 目录约定：评估用 best、训练恢复用 last（与 CMU 版一致）
    return f'./{ckpt}/ckpt_best.pth.tar' if opt.is_eval else f'./{ckpt}/ckpt_last.pth.tar'

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




def main(opt):
    """主训练函数（3DPW）"""
    lr_now = opt.lr_now
    start_epoch = 1

    print('>>> 创建模型')
    # MultiStage 模型（stage_4.MultiStageModel 与 h36m 的 BARNet 角色一致）
    net_pred = MultiStageModel_4(opt=opt)
    net_pred.to(opt.cuda_idx)

    # 创建分组优化器
    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, net_pred.parameters()),
        lr=opt.lr_now
    )
    print(f">>> 总参数量: {sum(p.numel() for p in net_pred.parameters()) / 1e6:.2f}M")

    # 加载检查点
    if opt.is_load or opt.is_eval:
        model_path = _resolve_ckpt_path(opt)
        print(f">>> 从 '{model_path}' 加载检查点")
        ckpt = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(f">>> 检查点加载完成 (epoch: {ckpt['epoch']} | err: {ckpt['err']})")

    # ===== 加载数据集 =====
    print('>>> 加载数据集')
    if not opt.is_eval:
        train_dataset = Datasets.Datasets(opt, split=0)
        print(f'>>> 训练集长度: {len(train_dataset)}')
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        valid_dataset = Datasets.Datasets(opt, split=1)
        print(f'>>> 验证集长度: {len(valid_dataset)}')
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # 覆盖 DIM_USED（由 3DPW 数据集提供）
        global DIM_USED
        DIM_USED = np.array(train_dataset.dim_used, dtype=np.int64)
    else:
        # 评估模式下也需要拿到 dim_used
        probe_dataset = Datasets.Datasets(opt, split=2)
        DIM_USED = np.array(probe_dataset.dim_used, dtype=np.int64)

    # 测试集
    test_dataset = Datasets.Datasets(opt, split=2)
    print(f'>>> 测试集长度: {len(test_dataset)}')
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ===== 评估模式 =====
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for key in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[key]])
            head = np.append(head, [key])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_3dpw')
        return

    # ===== 训练模式 =====
    err_best = 1000
    for epoch in range(start_epoch, opt.epoch + 1):
        is_best = False
        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))

        print(f'>>> 训练周期: {epoch}')
        # 训练
        ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=train_loader, opt=opt, epo=epoch)
        print(f'训练误差: {ret_train["m_p3d_pw"]:.3f}')

        # 验证
        ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epoch)
        print(f'验证误差: {ret_valid["m_p3d_pw"]:.3f}')

        # 测试（整体）
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epoch)
        last_test_key = list(ret_test.keys())[-1]
        print(f'测试误差: {ret_test[last_test_key]:.3f}')

        # 保存日志（与 main_h36m26 一致）
        ret_log = [epoch, lr_now]
        head = ['epoch', 'lr']
        for key in ret_train.keys():
            ret_log.append(ret_train[key])
            head.append(key)
        for key in ret_valid.keys():
            ret_log.append(ret_valid[key])
            head.append(f'valid_{key}')
        for key in ret_test.keys():
            ret_log.append(ret_test[key])
            head.append(f'test_{key}')
        log.save_csv_log(opt, head, ret_log, is_create=(epoch == 1))

        # 最佳模型
        if ret_valid['m_p3d_pw'] < err_best:
            err_best = ret_valid['m_p3d_pw']
            is_best = True

        # 保存检查点
        log.save_ckpt(
            {
                'epoch': epoch,
                'lr': lr_now,
                'err': ret_valid['m_p3d_pw'],
                'state_dict': net_pred.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            is_best=is_best,
            opt=opt
        )

        # 数值稳定性检查
        for name, param in net_pred.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}")
            if torch.isinf(param).any():
                print(f"Inf detected in {name}")


def eval(opt):
    print('>>> 创建模型')
    net_pred = MultiStageModel_4(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # 加载模型
    model_path = _resolve_ckpt_path(opt)
    print(f">>> 从 '{model_path}' 加载检查点")
    ckpt = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
    net_pred.load_state_dict(ckpt['state_dict'])
    print(f">>> 检查点加载完成 (epoch: {ckpt['epoch']} | err: {ckpt['err']})")

    # 准备数据加载器
    dataset = Datasets.Datasets(opt=opt, split=2)
    global DIM_USED
    DIM_USED = np.array(dataset.dim_used, dtype=np.int64)
    data_loader = DataLoader(
        dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 执行评估
    ret_test = run_model(net_pred, is_train=3, data_loader=data_loader, opt=opt)
    ret_log = np.array(['avg'])
    head = np.array(['set'])

    for key in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[key]])
        head = np.append(head, [f'test_{key}'])

    log.save_csv_eval_log(opt, head, ret_log, is_create=True)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=None, opt=None):
    net_pred.train() if is_train == 0 else net_pred.eval()
    dim_used = data_loader.dataset.dim_used
    l_p3d = 0
    if is_train <= 1:
        m_p3d_pw = 0
    else:
        titles = (np.arange(opt.output_n) + 1) * 40
        m_p3d_pw = np.zeros(opt.output_n)

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size 

    start_time = time.time()

    view_origin = None
    view_predict = None

    for i, (p3d_pw) in enumerate(data_loader):
        batch_size, seq_n, all_dim = p3d_pw.shape
        if batch_size == 1 and is_train == 0:
            continue

        n += batch_size
        batch_time = time.time()
        p3d_pw = p3d_pw.float().to(opt.cuda_idx)

        # 数据预处理器
        mix_vector = torch.tensor([[[0, 0.1, 0]]]).repeat(batch_size, seq_n, 1, 1).float().to(option.cuda_idx).expand(-1, -1, -1, 3)
        #对比实验：没有geo表征
        #geo = GEOMETRIC(leg_right=None, leg_left=None, fix=mix_vector,device = opt.cuda_idx)

        # 准备输入数据
        input_data = p3d_pw[:, :, DIM_USED].clone()
        # print(input_data[0,0,:])
        #对比实验：没有geo表征
        #input_data_geo = geo.encode(input_data, data_class='3dpw')
        p3d_sup = p3d_pw.clone()[:, :, DIM_USED][:, -out_n - seq_in:]

        #对比实验：没有geo表征
        #p3d_sup_geo = geo.encode(p3d_sup, data_class='3dpw')
        p3d_sup = p3d_sup.reshape([-1, seq_in + out_n, len(dim_used) // 3, 3])


        # 前向传播
        #对比实验：没有geo表征
        #outputs_geo = net_pred(input_data_geo)
        outputs = net_pred(input_data)
        # 数据还原
        p3d_out_all_fin, p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = outputs
        #对比实验：没有geo表征
        #p3d_out_all_fin_geo, p3d_out_all_4_geo, p3d_out_all_3_geo, p3d_out_all_2_geo, p3d_out_all_1_geo = outputs_geo
        # print(p3d_out_all_fin_geo[0,0,:])
        #对比实验：没有geo表征
        #p3d_out_all_fin = geo.decode(p3d_out_all_fin_geo, data_class='3dpw')
        # print(p3d_out_all_fin[0,0,:])
        # p3d_out_all_4 = geo.decode(p3d_out_all_4_geo)
        # p3d_out_all_3 = geo.decode(p3d_out_all_3_geo)
        # p3d_out_all_2 = geo.decode(p3d_out_all_2_geo)
        # p3d_out_all_1 = geo.decode(p3d_out_all_1_geo)

        # 处理输出
        p3d_out = p3d_pw.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all_fin[:, seq_in:]
        p3d_out = p3d_out.reshape([-1, out_n, all_dim // 3, 3])
        p3d_pw = p3d_pw.reshape([-1, in_n + out_n, all_dim // 3, 3])

        # 重塑输出
        #对比实验：没有geo表征
        #p3d_sup_geo = p3d_sup_geo.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        #p3d_out_all_fin_geo = p3d_out_all_fin_geo.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_fin = p3d_out_all_fin.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        # p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        # p3d_out_all_3 = p3d_out_all_3.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        # p3d_out_all_2 = p3d_out_all_2.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        # p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])

        # 训练步骤
        if is_train == 0:  # 计算损失
            # loss_p3d_fin = torch.mean(torch.norm(p3d_out_all_fin - p3d_sup, dim=3))
            # loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup, dim=3))
            # loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3 - p3d_sup, dim=3))
            # loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2 - p3d_sup, dim=3))
            # loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup, dim=3))
            
            #对比实验：没有geo表征
            #loss_p3d_fin_geo = torch.mean(torch.norm(p3d_out_all_fin_geo - p3d_sup_geo, dim=3))
            loss_p3d_fin = torch.mean(torch.norm(p3d_out_all_fin - p3d_sup, dim=3))

            # 组合损失
            # loss_all = loss_p3d_fin + loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1

            # 反向传播
            optimizer.zero_grad()
            loss_p3d_fin.backward()
            #对比实验：没有geo表征
            #loss_p3d_fin_geo.backward()
            nn.utils.clip_grad_norm_(net_pred.parameters(), max_norm=opt.max_norm)
            optimizer.step()

            # 更新日志
            l_p3d += loss_p3d_fin.cpu().data.numpy() * batch_size

        #记录骨架信息
        p3d_pw_cpu = p3d_pw.detach().cpu().numpy()
        p3d_out_cpu = p3d_out.detach().cpu().numpy()
        view_origin = p3d_pw_cpu[0, in_n:in_n + out_n, :, :]
        view_predict = p3d_out_cpu[0, :, :, :]

        # 计算误差
        if is_train <= 1:
            # 计算平均位置误差
            mpjpe = torch.mean(torch.norm(p3d_pw[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_pw += mpjpe.cpu().data.numpy() * batch_size
        else:
            # 计算每个时间步的位置误差
            mpjpe = torch.sum(torch.mean(torch.norm(p3d_pw[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_pw += mpjpe.cpu().data.numpy()


        # 打印进度
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            print(f'{i + 1}/{len(data_loader)} | '
                  f'批次时间: {time.time() - batch_time:.3f}s | '
                  f'单样本时间: {(time.time() - batch_time) / batch_size:.3f}s | '
                  f'总时间: {elapsed:.0f}s')
    #if is_train == 3:
        #view_gujia(view_origin, view_predict,name=f"epoch{epo}",data_class='3dpw')
        


        # 准备返回结果
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_pw"] = m_p3d_pw / n
    else:
        m_p3d_pw /= n
        for j in range(out_n):
            ret[f"#{titles[j]}ms"] = m_p3d_pw[j]

    return ret


if __name__ == '__main__':
    # 初始化配置和全局变量
    option = Options().parse()
    global_var._init()

    set_seed(334)

    # 执行主程序
    if option.is_eval:
        eval(option)
    else:
        main(option)
