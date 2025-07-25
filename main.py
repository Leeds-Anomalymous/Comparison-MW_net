import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm

# 导入自定义模块
from datasets import ImbalancedDataset
from model import ResNet32_1d, VNet
from evaluate import evaluate_model

def set_seed(seed):
    """设置随机种子以确保实验可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_var(x, requires_grad=True):
    """将张量转换为变量，处理CUDA移动"""
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.autograd.Variable(x, requires_grad=requires_grad)

def train_mwnet(model, vnet, train_loader, meta_loader, val_loader, test_loader, config, dataset_obj=None):
    """
    使用MW-Net算法训练模型
    
    Args:
        model: 要训练的主分类器模型
        vnet: Meta-Weight-Net模型
        train_loader: 训练数据加载器
        meta_loader: 元数据加载器（均衡）
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        config: 配置参数字典
        dataset_obj: 数据集对象(用于评估)
    
    Returns:
        训练好的模型
    """
    device = config['device']
    model.to(device)
    vnet.to(device)
    
    # 定义优化器
    optimizer_model = optim.SGD(model.parameters(), 
                               lr=config['learning_rate'],
                               momentum=0.9, 
                               weight_decay=5e-4)
    
    optimizer_vnet = optim.Adam(vnet.parameters(), 
                              lr=0.001,
                              weight_decay=1e-4)
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 训练日志
    train_losses = []
    meta_losses = []
    
    print(f"开始使用MW-Net训练 {config['model_type']} 模型 - 数据集: {config['dataset_name']}, 不平衡率: {config['rho']}")
    
    # 训练循环
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        vnet.train()
        
        train_loss = 0.0
        meta_loss = 0.0
        
        meta_loader_iter = iter(meta_loader)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 数据预处理
            if 'TBM' in config['dataset_name']:
                # 针对TBM数据的处理
                if len(inputs.shape) == 4:  # [batch_size, 1, 3, 1024]
                    inputs = inputs.squeeze(1)  # 移除额外的维度，变为[batch_size, 3, 1024]
                
                # 确保数据形状正确[batch_size, channels, length]
                if inputs.shape[1] != 3 and inputs.shape[2] == 3:
                    inputs = inputs.transpose(1, 2)  # 转换为[batch_size, channels, length]
                
                inputs = inputs.float().to(device)
            else:
                # 针对其他数据集的处理
                if len(inputs.shape) == 3:  # (N, 28, 28)
                    inputs = inputs.unsqueeze(1)  # 添加通道维度 -> (N, 1, 28, 28)
                # 修正通道顺序（如果需要）
                if inputs.shape[1] != 3 and inputs.shape[-1] == 3:
                    inputs = inputs.permute(0, 3, 1, 2)  # NHWC -> NCHW
                inputs = inputs.float().to(device)
            
            targets = targets.to(device)
            
            # 步骤1: 创建临时模型，拷贝当前模型参数
            meta_model = ResNet32_1d(input_channels=config['input_channels'], 
                                    seq_length=config['seq_length'], 
                                    num_classes=2).to(device)
            meta_model.load_state_dict(model.state_dict())
            
            # 步骤2: 计算每个样本的损失
            outputs = meta_model(inputs)
            cost = F.cross_entropy(outputs, targets, reduction='none')
            cost_v = cost.view(-1, 1)
            
            # 步骤3: 使用vnet计算样本权重
            with torch.no_grad():
                w_org = vnet(cost_v)
            
            # 步骤4: 使用权重计算加权损失
            loss = torch.sum(cost_v * w_org) / len(cost_v)
            
            # 步骤5: 计算元梯度
            meta_model.zero_grad()
            grads = torch.autograd.grad(loss, meta_model.parameters(), create_graph=True)
            
            # 步骤6: 进行虚拟更新
            meta_lr = config['learning_rate']
            meta_model_update = meta_model
            for i, (name, param) in enumerate(meta_model.named_parameters()):
                if param.requires_grad:
                    param.data = param.data - meta_lr * grads[i]
            
            # 步骤7: 在元数据集上计算损失
            try:
                meta_inputs, meta_targets = next(meta_loader_iter)
            except StopIteration:
                meta_loader_iter = iter(meta_loader)
                meta_inputs, meta_targets = next(meta_loader_iter)
            
            # 元数据预处理
            if 'TBM' in config['dataset_name']:
                if len(meta_inputs.shape) == 4:
                    meta_inputs = meta_inputs.squeeze(1)
                if meta_inputs.shape[1] != 3 and meta_inputs.shape[2] == 3:
                    meta_inputs = meta_inputs.transpose(1, 2)
                meta_inputs = meta_inputs.float().to(device)
            else:
                if len(meta_inputs.shape) == 3:
                    meta_inputs = meta_inputs.unsqueeze(1)
                if meta_inputs.shape[1] != 3 and meta_inputs.shape[-1] == 3:
                    meta_inputs = meta_inputs.permute(0, 3, 1, 2)
                meta_inputs = meta_inputs.float().to(device)
            
            meta_targets = meta_targets.to(device)
            
            # 使用更新后的元模型前向传播
            meta_outputs = meta_model_update(meta_inputs)
            meta_l = F.cross_entropy(meta_outputs, meta_targets)
            
            # 步骤8: 更新vnet参数
            optimizer_vnet.zero_grad()
            meta_l.backward()
            optimizer_vnet.step()
            
            # 步骤9: 计算带有更新后vnet的损失
            v_lambda = vnet(cost_v.detach())
            l_f = torch.sum(cost_v.detach() * v_lambda) / len(cost_v)
            
            # 步骤10: 更新主模型参数
            optimizer_model.zero_grad()
            l_f.backward()
            optimizer_model.step()
            
            # 记录损失
            train_loss += l_f.item()
            meta_loss += meta_l.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'meta_loss': meta_loss / (batch_idx + 1)
            })
        
        # 计算epoch平均损失
        epoch_loss = train_loss / len(train_loader)
        epoch_meta_loss = meta_loss / len(train_loader)
        train_losses.append(epoch_loss)
        meta_losses.append(epoch_meta_loss)
        
        print(f"Epoch {epoch} - 训练损失: {epoch_loss:.4f}, 元损失: {epoch_meta_loss:.4f}")
        
        # 在验证集上评估
        print(f"\n===== 在验证集上评估 Epoch {epoch} =====")
        val_metrics = evaluate_model(
            model, 
            val_loader, 
            save_dir=None,
            dataset_name=config['dataset_name'],
            rho=config['rho'],
            dataset_obj=dataset_obj,
            run_number=config['run_number'],
            model_type=config['model_type'],
            is_validation=True
        )
        
        # 在测试集上定期评估
        if epoch % config['eval_interval'] == 0 or epoch == config['epochs']:
            print(f"\n===== 在测试集上评估 Epoch {epoch}/{config['epochs']} =====")
            metrics = evaluate_model(
                model, 
                test_loader, 
                save_dir=None,
                dataset_name=config['dataset_name'],
                rho=config['rho'],
                dataset_obj=dataset_obj,
                run_number=config['run_number'],
                model_type=config['model_type']
            )
    
    # 保存最终模型
    model_filename = f"{config['dataset_name']}_{config['model_type']}_mwnet_rho{config['rho']}_run{config['run_number']}.pth"
    model_path = os.path.join(config['save_dir'], model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存到 {model_path}")
    
    # 保存VNet模型
    vnet_filename = f"{config['dataset_name']}_{config['model_type']}_vnet_rho{config['rho']}_run{config['run_number']}.pth"
    vnet_path = os.path.join(config['save_dir'], vnet_filename)
    torch.save(vnet.state_dict(), vnet_path)
    print(f"VNet模型已保存到 {vnet_path}")
    
    return model, vnet

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MW-Net训练和评估')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='TBM_K_M_Noise', help='数据集名称')
    parser.add_argument('--rho', type=float, default=0.01, help='不平衡因子(正类样本比例)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集占训练集的比例')
    parser.add_argument('--meta_ratio', type=float, default=0.1, help='元数据集占训练集的比例')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--eval_interval', type=int, default=5, help='测试集评估间隔(每多少个epoch评估一次)')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID，设为-1表示使用CPU')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 加载数据集
    print(f"正在加载 {args.dataset} 数据集, 不平衡率 rho={args.rho}")
    dataset = ImbalancedDataset(
        dataset_name=args.dataset,
        rho=args.rho,
        batch_size=args.batch_size,
        seed=args.seed,
        val_ratio=args.val_ratio,
        meta_ratio=args.meta_ratio
    )
    
    train_loader, val_loader, meta_loader, test_loader = dataset.get_dataloaders()
    
    # 打印数据集统计信息
    dist = dataset.get_class_distribution()
    print(f"训练集分布: 正类={dist['train'][0]}, 负类={dist['train'][1]}")
    print(f"验证集分布: 正类={dist['val'][0]}, 负类={dist['val'][1]}")
    print(f"元数据集分布: 正类={dist['meta'][0]}, 负类={dist['meta'][1]}")
    print(f"测试集分布: 正类={dist['test'][0]}, 负类={dist['test'][1]}")

    # 确定数据集的输入维度
    if 'TBM' in args.dataset:
        input_channels = 3
        seq_length = 1024  # TBM数据的序列长度
    elif args.dataset == 'cifar10':
        input_channels = 3
        seq_length = 32  # CIFAR-10的图像大小
    else:  # MNIST, Fashion-MNIST
        input_channels = 1
        seq_length = 28  # MNIST的图像大小
    
    # 运行两次实验
    for run_number in range(1, 3):
        print(f"\n{'='*50}")
        print(f"开始使用MW-Net训练 ResNet32_1d 模型在 {args.dataset} 数据集上 (第 {run_number} 次运行)")
        print(f"{'='*50}\n")
        
        # 创建配置字典
        config = {
            'dataset_name': args.dataset,
            'rho': args.rho,
            'batch_size': args.batch_size,
            'model_type': 'ResNet32_1d',
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'eval_interval': args.eval_interval,
            'seed': args.seed,
            'device': device,
            'save_dir': args.save_dir,
            'run_number': run_number,
            'input_channels': input_channels,
            'seq_length': seq_length
        }
        
        # 创建模型
        model = ResNet32_1d(input_channels=input_channels, seq_length=seq_length, num_classes=2)
        vnet = VNet(1, 100, 1)  # 输入维度为1，隐藏层100，输出维度为1
        
        # 训练模型
        print("\n开始MW-Net训练过程...")
        start_time = time.time()
        
        trained_model, trained_vnet = train_mwnet(
            model, vnet, train_loader, meta_loader, val_loader, test_loader, config, dataset_obj=dataset
        )
        
        # 打印训练时间
        training_time = time.time() - start_time
        print(f"\nMW-Net训练完成! 总用时: {training_time:.2f} 秒")
        
        # 最终评估
        print("\n进行最终评估...")
        metrics = evaluate_model(
            trained_model, 
            test_loader, 
            save_dir=config['save_dir'],
            dataset_name=config['dataset_name'],
            rho=config['rho'],
            dataset_obj=dataset,
            run_number=config['run_number'],
            model_type=f"{config['model_type']}_mwnet",
            is_final=True
        )
        
        print("\n===== 最终评估结果 =====")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # 释放内存
        del model, trained_model, vnet, trained_vnet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 释放数据集内存
    del dataset, train_loader, val_loader, meta_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    
#python main.py --dataset TBM_K_M_Noise --rho 0.01 --batch_size 64 --val_ratio 0.2 --meta_ratio 0.1 --epochs 100 --lr 0.01 --save_dir ./results