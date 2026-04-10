import sys
sys.path.append('../')
from utils.CMU_motion_3d import CMU_Motion3D
from utils.view import view_gujia
from model.model_h36m import MultiStageModel_4 as BARNet
#from thop import profile
from utils.geometric import GEOMETRIC
from utils.opt import Options
from utils import util
from utils import log
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import random
torch.backends.cudnn.enabled = False

# 设置环境变量解决CUDA错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义动作常量
ACTIONS = ["basketball", "basketball_signal", "directing_traffic", 
        "jumping", "running", "soccer", "walking", "washwindow"]

# 定义对称关节和需要忽略的关节
JOINT_TO_IGNORE = np.array([16, 20, 29, 24, 27, 33, 36])
JOINT_EQUAL = np.array([15, 15, 15, 23, 23, 32, 32])
VIEW_JOINT_USED = np.array([
        0,2,3,4,5,6,  8,9,10,11,12 , 14,15,17,18,19  ,21,22,23,25,26,28, 30,31,32,34,35,37
    ])

def set_seed(seed):
    """设置随机种子以确保实验可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(opt):
    """主训练函数"""
    set_seed(334)  # 设置随机种子
    
    lr_now = opt.lr_now
    start_epoch = 1
    print('>>> 创建模型')
    
    net_pred = BARNet(opt=opt)
    net_pred.to(opt.cuda_idx)

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

    print('>>> 加载数据集')
    if not opt.is_eval:
        # 训练集
        train_dataset = CMU_Motion3D(opt, split=0)
        print(f'>>> 训练集长度: {len(train_dataset)}')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=opt.batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=True
        )
        
        # 验证集
        valid_dataset = CMU_Motion3D(opt, split=2)
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
    for act in ACTIONS:
        test_dataset = CMU_Motion3D(opt=opt, split=2, actions=act)
        test_loader[act] = DataLoader(
            test_dataset, 
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
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, [k])
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
        print(f'训练误差: {ret_train["m_p3d_cmu"]:.3f}')
        
        # 验证步骤
        ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epoch)
        print(f'验证误差: {ret_valid["m_p3d_cmu"]:.3f}')

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
        ret_log = np.array([epoch, lr_now])
        head = np.array(['epoch', 'lr'])
        for k in ret_train.keys():
            ret_log = np.append(ret_log, [ret_train[k]])
            head = np.append(head, [k])
        for k in ret_valid.keys():
            ret_log = np.append(ret_log, [ret_valid[k]])
            head = np.append(head, ['valid_' + k])
        for k in ret_test.keys():
            ret_log = np.append(ret_log, [ret_test[k]])
            head = np.append(head, ['test_' + k])
        log.save_csv_log(opt, head, ret_log, is_create=(epoch == 1))

        # 更新最佳模型
        if ret_valid['m_p3d_cmu'] < err_best:
            err_best = ret_valid['m_p3d_cmu']
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

        log.save_ckpt({'epoch': epoch,
                    'lr': lr_now,
                    'err': ret_valid['m_p3d_cmu'],
                    'state_dict': net_pred.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    is_best=is_best, opt=opt)


def eval(opt):
    """模型评估函数"""
    set_seed(334)  # 设置随机种子
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
    for act in ACTIONS:
        dataset = CMU_Motion3D(opt=opt, split=2, actions=act)
        data_loader[act] = DataLoader(
            dataset, 
            batch_size=opt.test_batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )

    # 验证集
    valid_dataset = CMU_Motion3D(opt, split=2)
    print(f'>>> 验证集长度: {len(valid_dataset)}')
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=opt.test_batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    # 执行验证
    ret_valid = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt)
    print(f'验证误差: {ret_valid["m_p3d_cmu"]:.3f}')
    
    # 执行测试
    is_create = True
    avg_ret_log = []
    test_error = 0
    
    for act in ACTIONS:
        ret_test = run_model(net_pred, is_train=3, data_loader=data_loader[act], opt=opt)
        
        # 准备日志数据
        ret_log = [act]
        head = ['action']
        for k in ret_test.keys():
            ret_log.append(ret_test[k])
            head.append(f'test_{k}')
            if k.startswith("#") and "ms" in k:
                test_error += ret_test[k]
                
        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    # 计算并保存平均结果
    test_error /= (25 * len(ACTIONS))
    print(f'测试误差: {test_error:.3f}')
    
    avg_ret_log = np.array(avg_ret_log, dtype=np.float64).mean(axis=0)
    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=None, opt=None):
    """
        运行模型训练/评估
        is_train: 0-训练, 1-验证, 3-测试
        """
    # 设置模型模式
    net_pred.train() if is_train == 0 else net_pred.eval()

    # 初始化变量
    l_p3d = 0
    if is_train <= 1:
        m_p3d_cmu = 0
    else:
        titles = (np.arange(opt.output_n) + 1) * 40
        m_p3d_cmu = np.zeros(opt.output_n)

    n = 0
    in_n = opt.input_n
    out_n = opt.output_n
    seq_in = opt.kernel_size
    start_time = time.time()
    # 计算对称关节索引
    index_to_ignore = np.concatenate((JOINT_TO_IGNORE * 3, JOINT_TO_IGNORE * 3 + 1, JOINT_TO_IGNORE * 3 + 2))
    index_to_equal = np.concatenate((JOINT_EQUAL * 3, JOINT_EQUAL * 3 + 1, JOINT_EQUAL * 3 + 2))
    # 遍历数据加载器
    for x, (p3d_cmu) in enumerate(data_loader):
        # print(i)
        batch_size, seq_n, all_dim = p3d_cmu.shape
        # 跳过单样本批次（训练时）
        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        batch_time = time.time()
        p3d_cmu = p3d_cmu.float().to(option.cuda_idx)

        # 获取使用的维度
        dim_used = data_loader.dataset.dim_used

        # 准备输入数据
        input_data = p3d_cmu[:, :, dim_used].clone()
        # 用于可视化部分

        # p3d_h36_t = p3d_cmu[:, :, dim_used].clone()  # (bs,seq,78)
        # num11 = p3d_cmu[:, :, [33, 34, 35]].unsqueeze(2)
        # num1 = p3d_cmu[:, :, [3, 4, 5]].unsqueeze(2)
        # num6 = p3d_cmu[:, :, [18, 19, 20]].unsqueeze(2)

        # 数据预处理器
        node2 = p3d_cmu[:, :, [6, 7, 8]].unsqueeze(2)
        # print(leg_right)
        node8 = p3d_cmu[:, :, [24, 25, 26]].unsqueeze(2)
        # print(leg_left)
        values = torch.tensor([[[0, 0.1, 0]]]).repeat(batch_size, seq_n, 1, 1).float().to(option.cuda_idx)
        mixed_tensor = values.expand(-1, -1, -1, 3) #(bs,seq,1,3)

        # print(mixed_tensor)
        geo = GEOMETRIC(leg_right=node2, leg_left=node8, fix=mixed_tensor,device = opt.cuda_idx)

        input_data_geo = geo.encode(input_data, data_class='cmu')
        p3d_sup = p3d_cmu.clone()[:, :, dim_used][:, -out_n - seq_in:]
        p3d_sup_geo = geo.encode(p3d_sup, data_class='cmu')
        p3d_sup = p3d_sup.reshape([-1, seq_in + out_n, len(dim_used) // 3, 3])

        # 参数大小和flops的评估
        #flops, params = profile(net_pred, inputs=(input_data_geo,), verbose=False)
        # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
        # print(f"Params: {params / 1e6:.2f} MParams")

        # 前向传播
        outputs_geo = net_pred(input_data_geo)
        # outputs = net_pred(input_data)
        # 数据还原
        # p3d_out_all_fin, p3d_out_all_4, p3d_out_all_3, p3d_out_all_2, p3d_out_all_1 = outputs
        p3d_out_all_fin_geo, p3d_out_all_4_geo, p3d_out_all_3_geo, p3d_out_all_2_geo, p3d_out_all_1_geo = outputs_geo
        # print(p3d_out_all_fin_geo[0,0,:])
        p3d_out_all_fin = geo.decode(p3d_out_all_fin_geo,data_class='cmu')
        # print(p3d_out_all_fin[0,0,:])
        # p3d_out_all_4 = geo.decode(p3d_out_all_4_geo)
        # p3d_out_all_3 = geo.decode(p3d_out_all_3_geo)
        # p3d_out_all_2 = geo.decode(p3d_out_all_2_geo)
        # p3d_out_all_1 = geo.decode(p3d_out_all_1_geo)

        # 处理输出
        p3d_out = p3d_cmu.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all_fin[:, seq_in:]
        p3d_out[:, :, index_to_ignore] = p3d_out[:, :, index_to_equal]
        p3d_out = p3d_out.reshape([-1, out_n, all_dim // 3, 3])
        p3d_cmu = p3d_cmu.reshape([-1, in_n + out_n, all_dim // 3, 3])

        # 重塑输出
        p3d_sup_geo = p3d_sup_geo.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_out_all_fin_geo = p3d_out_all_fin_geo.reshape([batch_size, seq_in + out_n, len(dim_used) // 3, 3])
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

            loss_p3d_fin_geo = torch.mean(torch.norm(p3d_out_all_fin_geo - p3d_sup_geo, dim=3))
            loss_p3d_fin = torch.mean(torch.norm(p3d_out_all_fin - p3d_sup, dim=3))

            p3d_h36_cpu = p3d_cmu.detach().cpu().numpy()
            p3d_out_cpu = p3d_out.detach().cpu().numpy()
            #view_gujia(p3d_h36_cpu[0,in_n:in_n + out_n,VIEW_JOINT_USED,:].transpose(1,0,2), p3d_out_cpu[0,in_n:in_n + out_n,VIEW_JOINT_USED,:].transpose(1,0,2),data_class='cmu')  #可以考虑使用部分关节点

            # 组合损失
            # loss_all = loss_p3d_fin + loss_p3d_4 + loss_p3d_3 + loss_p3d_2 + loss_p3d_1

            # 反向传播
            optimizer.zero_grad()
            # loss_p3d_fin.backward()
            loss_p3d_fin_geo.backward()
            nn.utils.clip_grad_norm_(net_pred.parameters(), max_norm=opt.max_norm)
            optimizer.step()

            # 更新日志
            l_p3d += loss_p3d_fin.cpu().data.numpy() * batch_size

        # 计算误差
        if is_train <= 1:
            # 计算平均位置误差
            mpjpe = torch.mean(torch.norm(p3d_cmu[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_cmu += mpjpe.cpu().data.numpy() * batch_size
        else:
            # 计算每个时间步的位置误差
            mpjpe = torch.sum(torch.mean(torch.norm(p3d_cmu[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_cmu += mpjpe.cpu().data.numpy()

        # 打印进度
        if x % 1000 == 0:
            elapsed = time.time() - start_time
            print(f'{x + 1}/{len(data_loader)} | '
                  f'批次时间: {time.time() - batch_time:.3f}s | '
                  f'单样本时间: {(time.time() - batch_time) / batch_size:.3f}s | '
                  f'总时间: {elapsed:.0f}s')

    # 准备返回结果
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_cmu"] = m_p3d_cmu / n
    else:
        m_p3d_cmu /= n
        for j in range(out_n):
            ret[f"#{titles[j]}ms"] = m_p3d_cmu[j]

    return ret



if __name__ == '__main__':
    # 初始化配置
    option = Options().parse()
    
    # 执行主程序
    if option.is_eval:
        eval(option)
    else:
        main(option)