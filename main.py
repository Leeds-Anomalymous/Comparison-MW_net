import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import copy
from datetime import datetime
from tqdm import tqdm

# 导入自定义模块
from datasets import ImbalancedDataset
from model import ResNet32_1d, BiLSTM, create_transformer
from evaluate import evaluate_model

def set_seed(seed):
    """设置随机种子以确保实验可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, test_loader, config, dataset_obj=None):
    """
    训练模型函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        config: 配置参数字典
        dataset_obj: 数据集对象(用于评估)
    
    Returns:
        训练好的模型
    """
    device = config['device']
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 训练日志
    train_losses = []
    val_losses = []
    
    # 早停参数
    best_val_metric = 0.0  # 使用验证集G-mean作为早停指标
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = config['patience']
    counter = 0
    early_stopped = False
    
    print(f"开始训练 {config['model_type']} 模型 - 数据集: {config['dataset_name']}, 不平衡率: {config['rho']}")
    
    # 训练循环
    for epoch in range(1, config['epochs'] + 1):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        for data, labels in progress_bar:
            # 数据预处理
            if 'TBM' in config['dataset_name'] or config['model_type'] == 'ResNet32_1d':
                # 针对TBM数据或1D模型的处理
                if len(data.shape) == 4:  # [batch_size, 1, 3, 1024]
                    data = data.squeeze(1)  # 移除额外的维度，变为[batch_size, 3, 1024]
                
                # 确保数据形状正确[batch_size, channels, length]
                if data.shape[1] != 3 and data.shape[2] == 3:
                    data = data.transpose(1, 2)  # 转换为[batch_size, channels, length]
                
                data = data.float().to(device)
            else:
                # 图像数据需要添加通道维度
                if len(data.shape) == 3:  # (N, 28, 28)
                    data = data.unsqueeze(1)  # 添加通道维度 -> (N, 1, 28, 28)
                # 修正通道顺序（如果需要）
                if data.shape[1] != 3 and data.shape[-1] == 3:
                    data = data.permute(0, 3, 1, 2)  # NHWC -> NCHW
                data = data.float().to(device)
            
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        # 计算epoch平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        
        print(f"Epoch {epoch} - 训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.2f}%")
        
        # 每个epoch都在验证集上评估
        print(f"\n===== 在验证集上评估 Epoch {epoch} =====")
        val_metrics = evaluate_model(
            model, 
            val_loader, 
            save_dir=None,  # 不保存验证集的混淆矩阵
            dataset_name=config['dataset_name'],
            rho=config['rho'],
            dataset_obj=dataset_obj,
            run_number=config['run_number'],
            model_type=config['model_type'],
            is_validation=True  # 标记这是验证集评估
        )
        
        val_losses.append(val_metrics['g_mean'])  # 使用G-mean作为监控指标
        
        # 早停检查
        current_val_metric = val_metrics['g_mean']
        if current_val_metric > best_val_metric:
            print(f"验证集G-mean提升: {best_val_metric:.4f} -> {current_val_metric:.4f}")
            best_val_metric = current_val_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            print(f"验证集G-mean未提升, 当前耐心: {counter}/{patience}")
            if counter >= patience:
                print(f"早停触发! 已连续 {patience} 个epoch未见改善")
                early_stopped = True
                break
        
        # 在测试集上定期评估，但不保存混淆矩阵
        if epoch % config['eval_interval'] == 0 or epoch == config['epochs']:
            print(f"\n===== 在测试集上评估 Epoch {epoch}/{config['epochs']} =====")
            metrics = evaluate_model(
                model, 
                test_loader, 
                save_dir=None,  # 中间评估不保存混淆矩阵
                dataset_name=config['dataset_name'],
                rho=config['rho'],
                dataset_obj=dataset_obj,
                run_number=config['run_number'],
                model_type=config['model_type']
            )
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print(f"\n训练{'已提前结束' if early_stopped else '已完成'}, 使用验证集上最佳性能的模型")
    
    # 保存最佳模型
    model_filename = f"{config['dataset_name']}_{config['model_type']}_rho{config['rho']}_run{config['run_number']}.pth"
    model_path = os.path.join(config['save_dir'], model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"最佳模型已保存到 {model_path}")
    
    return model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型训练和评估')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'cifar10', 'fashion_mnist', 'TBM_K', 'TBM_M', 'TBM_K_M', 
                                 'TBM_K_Noise', 'TBM_M_Noise', 'TBM_K_M_Noise',
                                 'TBM_K_M_Noise_snr_3', 'TBM_K_M_Noise_snr_1', 'TBM_K_M_Noise_snr_0',
                                 'TBM_K_M_Noise_snr_-1', 'TBM_K_M_Noise_snr_-3', 'TBM_K_M_Noise_snr_-5',
                                 'TBM_K_M_Noise_snr_-7', 'TBM_K_M_Noise_snr_-10'],
                        help='数据集名称')
    parser.add_argument('--rho', type=float, default=0.01, help='不平衡因子(正类样本比例)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集占训练集的比例')
    
    # 训练参数
    parser.add_argument('--eval_interval', type=int, default=5, help='测试集评估间隔(每多少个epoch评估一次)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID，设为-1表示使用CPU')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值，连续多少个epoch无改善后停止')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    
    # 添加新参数控制是否遍历SNR数据集
    parser.add_argument('--run_snr_experiments', action='store_true', 
                        help='是否运行SNR实验，如果设置则遍历所有SNR数据集')
    
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
    
    # 创建模型训练参数字典 - 更新为推荐的epochs值
    models_config = {
        'ResNet32_1d': {
            'learning_rate': 0.001,
        },
        'BiLSTM': {
            'learning_rate': 0.001,
        },
        'Transformer': {
            'learning_rate': 0.0005,  # Transformer通常使用更小的学习率
        }
    }
    
    # 如果设置了运行SNR实验，则遍历所有SNR数据集
    if args.run_snr_experiments:
        snr_datasets = [
            'TBM_K_M_Noise_snr_3',
            'TBM_K_M_Noise_snr_1', 
            'TBM_K_M_Noise_snr_0',
            'TBM_K_M_Noise_snr_-1',
            'TBM_K_M_Noise_snr_-3',
            'TBM_K_M_Noise_snr_-5',
            'TBM_K_M_Noise_snr_-7',
            'TBM_K_M_Noise_snr_-10'
        ]
        datasets_to_run = snr_datasets
    else:
        # 只运行指定的数据集
        datasets_to_run = [args.dataset]
    
    # 外层循环：遍历所有模型
    for model_type, model_params in models_config.items():
        print(f"\n{'='*80}")
        print(f"开始训练 {model_type} 模型")
        print(f"{'='*80}\n")
        
        # 内层循环：遍历所有数据集
        for dataset_name in datasets_to_run:
            print(f"\n{'-'*60}")
            print(f"当前数据集: {dataset_name}")
            print(f"{'-'*60}\n")
            
            # 加载数据集
            print(f"正在加载 {dataset_name} 数据集, 不平衡率 rho={args.rho}")
            dataset = ImbalancedDataset(
                dataset_name=dataset_name,
                rho=args.rho,
                batch_size=args.batch_size,
                seed=args.seed,
                val_ratio=args.val_ratio
            )
            
            train_loader, val_loader, test_loader = dataset.get_dataloaders()
            
            # 打印数据集统计信息
            dist = dataset.get_class_distribution()
            print(f"训练集分布: 正类={dist['train'][0]}, 负类={dist['train'][1]}")
            print(f"验证集分布: 正类={dist['val'][0]}, 负类={dist['val'][1]}")  
            print(f"测试集分布: 正类={dist['test'][0]}, 负类={dist['test'][1]}")

            # 确定数据集的输入维度
            if 'TBM' in dataset_name:
                input_channels = 3
                seq_length = 1024  # TBM数据的序列长度
            elif dataset_name == 'cifar10':
                input_channels = 3
                seq_length = 32  # CIFAR-10的图像大小
            else:  # MNIST, Fashion-MNIST
                input_channels = 1
                seq_length = 28  # MNIST的图像大小
            
            # 每个模型在每个数据集上训练两次
            for run_number in range(1, 3):
                print(f"\n{'='*50}")
                print(f"开始训练 {model_type} 模型在 {dataset_name} 数据集上 (第 {run_number} 次运行)")
                print(f"{'='*50}\n")
                
                # 创建配置字典
                config = {
                    'dataset_name': dataset_name,
                    'rho': args.rho,
                    'batch_size': args.batch_size,
                    'model_type': model_type,
                    'learning_rate': model_params['learning_rate'],
                    'epochs': get_recommended_epochs(dataset_name, model_type, args.rho),
                    'eval_interval': args.eval_interval,
                    'seed': args.seed,
                    'device': device,
                    'save_dir': args.save_dir,
                    'run_number': run_number,
                    'patience': args.patience
                }
                
                # 创建对应的模型
                if model_type == 'ResNet32_1d':
                    model = ResNet32_1d(input_channels=input_channels, seq_length=seq_length, num_classes=2)
                elif model_type == 'BiLSTM':
                    if 'TBM' in dataset_name:
                        # BiLSTM需要输入为(batch, seq_len, features)
                        input_size = input_channels  # 特征数为通道数
                        hidden_size = 128
                        num_layers = 2
                        model = BiLSTM(input_size, hidden_size, num_layers, num_classes=2)
                    else:
                        # 对于图像数据，把它当作一个序列
                        input_size = seq_length  # 每一行作为一个时间步的特征
                        hidden_size = 128
                        num_layers = 2
                        model = BiLSTM(input_size, hidden_size, num_layers, num_classes=2)
                elif model_type == 'Transformer':
                    model = create_transformer(input_dim=input_channels, seq_length=seq_length, num_classes=2)
                else:
                    raise ValueError(f"不支持的模型类型: {model_type}")
                
                # 训练模型
                print("\n开始训练过程...")
                start_time = time.time()
                
                trained_model = train_model(model, train_loader, val_loader, test_loader, config, dataset_obj=dataset)
                
                # 打印训练时间
                training_time = time.time() - start_time
                print(f"\n训练完成! 总用时: {training_time:.2f} 秒")
                
                # 最终评估 - 只在这里保存混淆矩阵
                print("\n进行最终评估...")
                metrics = evaluate_model(
                    trained_model, 
                    test_loader, 
                    save_dir=config['save_dir'],  # 只在最终评估保存混淆矩阵
                    dataset_name=config['dataset_name'],
                    rho=config['rho'],
                    dataset_obj=dataset,
                    run_number=config['run_number'],
                    model_type=config['model_type'],
                    is_final=True  # 标记这是最终评估
                )
                
                print("\n===== 最终评估结果 =====")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")
                
                # 释放内存
                del model, trained_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 释放数据集内存
            del dataset, train_loader, val_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def get_recommended_epochs(dataset_name, model_type, rho):
    """
    根据数据集、模型类型和不平衡率返回推荐的训练轮数
    
    Args:
        dataset_name: 数据集名称
        model_type: 模型类型
        rho: 不平衡率
    
    Returns:
        推荐的训练轮数
    """
    # 基础轮数配置
    base_epochs = {
        # 简单数据集配置
        'mnist': {
            'ResNet32_1d': 25,
            'BiLSTM': 30,
            'Transformer': 35,
        },
        'fashion_mnist': {
            'ResNet32_1d': 30,
            'BiLSTM': 35,
            'Transformer': 40,
        },
        # 复杂数据集配置
        'cifar10': {
            'ResNet32_1d': 50,
            'BiLSTM': 60,
            'Transformer': 70,
        },
        # TBM数据集配置
        'TBM_K': {
            'ResNet32_1d': 50,
            'BiLSTM': 60,
            'Transformer': 80,
        },
        'TBM_M': {
            'ResNet32_1d': 50,
            'BiLSTM': 60,
            'Transformer': 80,
        },
        'TBM_K_M': {
            'ResNet32_1d': 60,
            'BiLSTM': 70,
            'Transformer': 90,
        },
        # 带噪声的TBM数据集可能需要更多轮次
        'TBM_K_Noise': {
            'ResNet32_1d': 70,
            'BiLSTM': 80,
            'Transformer': 100,
        },
        'TBM_M_Noise': {
            'ResNet32_1d': 70,
            'BiLSTM': 80,
            'Transformer': 100,
        },
        'TBM_K_M_Noise': {
            'ResNet32_1d': 80,
            'BiLSTM': 90,
            'Transformer': 120,
        },
    }
    
    # 为所有SNR变体数据集添加配置
    snr_datasets = {
        'TBM_K_M_Noise_snr_3': {
            'ResNet32_1d': 80,
            'BiLSTM': 90,
            'Transformer': 120,
        },
        'TBM_K_M_Noise_snr_1': {
            'ResNet32_1d': 85,
            'BiLSTM': 95,
            'Transformer': 125,
        },
        'TBM_K_M_Noise_snr_0': {
            'ResNet32_1d': 90,
            'BiLSTM': 100,
            'Transformer': 130,
        },
        'TBM_K_M_Noise_snr_-1': {
            'ResNet32_1d': 95,
            'BiLSTM': 105,
            'Transformer': 135,
        },
        'TBM_K_M_Noise_snr_-3': {
            'ResNet32_1d': 100,
            'BiLSTM': 110,
            'Transformer': 140,
        },
        'TBM_K_M_Noise_snr_-5': {
            'ResNet32_1d': 105,
            'BiLSTM': 115,
            'Transformer': 145,
        },
        'TBM_K_M_Noise_snr_-7': {
            'ResNet32_1d': 110,
            'BiLSTM': 120,
            'Transformer': 150,
        },
        'TBM_K_M_Noise_snr_-10': {
            'ResNet32_1d': 120,
            'BiLSTM': 130,
            'Transformer': 160,
        },
    }
    
    # 合并所有数据集配置
    base_epochs.update(snr_datasets)
    
    # 获取基础轮数
    if dataset_name in base_epochs and model_type in base_epochs[dataset_name]:
        epochs = base_epochs[dataset_name][model_type]
    else:
        # 默认值
        print(f"警告: 未找到 {dataset_name}/{model_type} 的推荐轮数，使用默认值50")
        epochs = 50
    
    # 根据不平衡率调整轮数
    # 当不平衡率较低时，增加轮数以更好地学习少数类特征
    if rho <= 0.01:
        epochs = int(epochs * 1.3)  # 不平衡率很低时增加30%
    elif rho <= 0.05:
        epochs = int(epochs * 1.2)  # 不平衡率较低时增加20%
    elif rho <= 0.1:
        epochs = int(epochs * 1.1)  # 不平衡率中等时增加10%
    
    print(f"推荐的 {dataset_name} 数据集上 {model_type} 模型的训练轮数: {epochs} (不平衡率 rho={rho})")
    return epochs

if __name__ == "__main__":
    main()
# python main.py --dataset TBM_K_M --rho 0.01 --val_ratio 0.2 --patience 10 --save_dir /root/autodl-tmp/results
# python /workspace/RL/baseline/main.py --run_snr_experiments --rho 0.01 --val_ratio 0.2 --patience 10 --save_dir /workspace/RL/baseline/results