import os
import time
import sys
sys.path.append('./')
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import h36motion3d as datasets
from utils import util, log, global_var
from utils.opt import Options
from model.model_h36m import MultiStageModel_4 as BARNet
import random

# 设置环境变量解决CUDA错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义动作常量
ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog",
    "walkingtogether"
]

# 定义使用的维度索引
DIM_USED = np.array([
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92
])

# 定义需要忽略的关节和对应的索引
JOINT_TO_IGNORE = np.array([16, 20, 23, 24, 28, 31])
INDEX_TO_IGNORE = np.concatenate((
    JOINT_TO_IGNORE * 3, 
    JOINT_TO_IGNORE * 3 + 1, 
    JOINT_TO_IGNORE * 3 + 2
))

# 定义对称关节和对应的索引
JOINT_EQUAL = np.array([13, 19, 22, 13, 27, 30])
INDEX_TO_EQUAL = np.concatenate((
    JOINT_EQUAL * 3, 
    JOINT_EQUAL * 3 + 1, 
    JOINT_EQUAL * 3 + 2
))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(opt):
    """主训练函数"""
    lr_now = opt.lr_now
    start_epoch = 1
    
    print('>>> 创建模型')
    net_pred = BARNet(opt=opt)
    net_pred.to(opt.cuda_idx)
    
    # 初始化优化器
    optimizer = optim.Adam(
        filter(lambda x: x.requires_grad, net_pred.parameters()), 
        lr=opt.lr_now
    )
    print(f">>> 总参数量: {sum(p.numel() for p in net_pred.parameters()) / 1e6:.2f}M")

    # 加载检查点
    if opt.is_load or opt.is_eval:
        model_path = './{}/ckpt_best.pth.tar'.format(opt.ckpt) if opt.is_eval else './{}/ckpt_last.pth.tar'.format(opt.ckpt)
        print(f">>> 从 '{model_path}' 加载检查点")
        ckpt = torch.load(model_path)
        start_epoch = ckpt['epoch'] + 1
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        net_pred.load_state_dict(ckpt['state_dict'])
        print(f">>> 检查点加载完成 (epoch: {ckpt['epoch']} | err: {ckpt['err']})")

    # 加载数据集
    print('>>> 加载数据集')
    if not opt.is_eval:
        # 训练集
        train_dataset = datasets.Datasets(opt, split=0)
        print(f'>>> 训练集长度: {len(train_dataset)}')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        
        # 验证集
        valid_dataset = datasets.Datasets(opt, split=2)
        print(f'>>> 验证集长度: {len(valid_dataset)}')
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=opt.test_batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=True
        )

    # 测试集
    test_loader = {}
    for action in ACTIONS:
        dataset = datasets.Datasets(opt=opt, split=2, actions=action)
        test_loader[action] = DataLoader(
            dataset, 
            batch_size=opt.test_batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )

    # 评估模式
    if opt.is_eval:
        ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
        ret_log = np.array([])
        head = np.array([])
        for key in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[key]])
            head = np.append(head, [key])
        log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
        return

    # 训练模式
    err_best = 1000
    for epoch in range(start_epoch, opt.epoch + 1):
        is_best = False
        lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / opt.epoch))
        
        print(f'>>> 训练周期: {epoch}')
        # 训练步骤
        ret_train = run_model(net_pred, optimizer, is_train=0, data_loader=train_loader, opt=opt, epo=epoch)
        print(f'训练误差: {ret_train["m_p3d_h36"]:.3f}')
        
        # 验证步骤
        ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epoch)
        print(f'验证误差: {ret_valid["m_p3d_h36"]:.3f}')

        # 测试步骤
        eval_results = []
        avg_ret_log = []
        for action in ACTIONS:
            ret_test = run_model(net_pred, is_train=3, data_loader=test_loader[action], opt=opt, epo=epoch)
            
            # 准备日志数据
            ret_log_eval = [action]
            head_eval = ['action']
            for key in ret_test.keys():
                ret_log_eval.append(ret_test[key])
                head_eval.append(f'test_{key}')
                
            eval_results.append(ret_log_eval)
            avg_ret_log.append(ret_log_eval[1:])

        # 计算平均误差
        avg_ret_log = np.array(avg_ret_log, dtype=np.float64).mean(axis=0)
        last_test_key = list(ret_test.keys())[-1]
        print(f'测试误差: {ret_test[last_test_key]:.3f}')

        # 保存日志和检查点
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
        
        # 更新最佳模型
        if ret_valid['m_p3d_h36'] < err_best:
            err_best = ret_valid['m_p3d_h36']
            is_best = True
            
            # 保存评估结果
            is_create = True
            for i, action in enumerate(ACTIONS):
                log.save_csv_eval_log(opt, head_eval, np.array(eval_results[i]), is_create=is_create)
                is_create = False
                
            # 保存平均结果
            write_ret_log = eval_results[0].copy()
            write_ret_log[0] = 'avg'
            write_ret_log[1:] = avg_ret_log
            log.save_csv_eval_log(opt, head_eval, write_ret_log, is_create=False)

        # 保存检查点
        log.save_ckpt(
            {
                'epoch': epoch,
                'lr': lr_now,
                'err': ret_valid['m_p3d_h36'],
                'state_dict': net_pred.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            is_best=is_best, 
            opt=opt
        )


def eval(opt):
    """模型评估函数"""
    print('>>> 创建模型')
    net_pred = BARNet(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # 加载模型
    model_path = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
    print(f">>> 从 '{model_path}' 加载检查点")
    ckpt = torch.load(model_path, map_location='cpu')
    net_pred.load_state_dict(ckpt['state_dict'])
    print(f">>> 检查点加载完成 (epoch: {ckpt['epoch']} | err: {ckpt['err']})")

    # 准备数据加载器
    data_loader = {}
    for action in ACTIONS:
        dataset = datasets.Datasets(opt=opt, split=2, actions=action)
        data_loader[action] = DataLoader(
            dataset, 
            batch_size=opt.test_batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )

    # 执行评估
    is_create = True
    avg_ret_log = []
    for action in ACTIONS:
        ret_test = run_model(net_pred, is_train=3, data_loader=data_loader[action], opt=opt, action=action)
        
        # 准备日志数据
        ret_log = [action]
        head = ['action']
        for key in ret_test.keys():
            ret_log.append(ret_test[key])
            head.append(f'test_{key}')
            
        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    # 保存平均结果
    avg_ret_log = np.array(avg_ret_log, dtype=np.float64).mean(axis=0)
    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=None, opt=None, action=None):
    """
    运行模型训练/评估
    is_train: 0-训练, 1-验证, 3-测试 
    """
    # 设置模型模式
    net_pred.train() if is_train == 0 else net_pred.eval()

    # 初始化变量
    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.arange(opt.output_n) + 1) * 40
        m_p3d_h36 = np.zeros(opt.output_n)
        
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    start_time = time.time()

    # 遍历数据加载器
    for i, (p3d_h36) in enumerate(data_loader):
        batch_size, seq_n, all_dim = p3d_h36.shape
        # 跳过单样本批次（训练时）
        if batch_size == 1 and is_train == 0:
            continue
            
        n += batch_size
        batch_time = time.time()
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)

        # 准备输入数据
        input_data = p3d_h36[:, :, DIM_USED].clone()
        p3d_sup = p3d_h36.clone()[:, :, DIM_USED][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(DIM_USED) // 3, 3]
        )

        # 前向传播
        outputs = net_pred(input_data)
        p3d_out_all_fin, p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = outputs

        # 处理输出
        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, DIM_USED] = p3d_out_all_fin[:, seq_in:]
        p3d_out[:, :, INDEX_TO_IGNORE] = p3d_out[:, :, INDEX_TO_EQUAL]
        p3d_out = p3d_out.reshape([-1, out_n, all_dim // 3, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim // 3, 3])

        # 重塑输出
        p3d_out_all_fin = p3d_out_all_fin.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        p3d_out_all_4 = p3d_out_all_4.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        p3d_out_all_3 = p3d_out_all_3.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        p3d_out_all_2 = p3d_out_all_2.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])
        p3d_out_all_1 = p3d_out_all_1.reshape([batch_size, seq_in + out_n, len(DIM_USED) // 3, 3])

        # 训练步骤
        if is_train == 0:            # 计算损失
            loss_p3d_fin = torch.mean(torch.norm(p3d_out_all_fin - p3d_sup, dim=3))
            loss_p3d_4 = torch.mean(torch.norm(p3d_out_all_4 - p3d_sup, dim=3))
            loss_p3d_3 = torch.mean(torch.norm(p3d_out_all_3 - p3d_sup, dim=3))
            loss_p3d_2 = torch.mean(torch.norm(p3d_out_all_2 - p3d_sup, dim=3))
            loss_p3d_1 = torch.mean(torch.norm(p3d_out_all_1 - p3d_sup, dim=3))
            
            # 组合损失
            loss_all = loss_p3d_fin + loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1

            # 反向传播
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(net_pred.parameters(), max_norm=opt.max_norm)
            optimizer.step()
            
            # 更新日志
            l_p3d += loss_p3d_fin.cpu().data.numpy() * batch_size

        # 计算误差
        if is_train <= 1:
            # 计算平均位置误差
            mpjpe = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe.cpu().data.numpy() * batch_size
        else:
            # 计算每个时间步的位置误差
            mpjpe = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe.cpu().data.numpy()
            
        # 打印进度
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            print(f'{i+1}/{len(data_loader)} | '
                  f'批次时间: {time.time() - batch_time:.3f}s | '
                  f'总时间: {elapsed:.0f}s')

    # 准备返回结果
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
        
    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 /= n
        for j in range(out_n):
            ret[f"#{titles[j]}ms"] = m_p3d_h36[j]
            
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