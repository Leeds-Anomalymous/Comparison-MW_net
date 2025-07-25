import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channels=3, seq_length=32, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.seq_length = seq_length // 4 
        self.avgpool = nn.AdaptiveAvgPool1d(self.seq_length)
        self.fc = nn.Linear(64 * self.seq_length, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet32_1d(input_channels=3, seq_length=32, num_classes=10):
    """创建自定义的ResNet-32模型"""
    return ResNet(BasicBlock, [5, 5, 5], input_channels, seq_length, num_classes)    


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, bidirectional=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 因为双向
        
    def forward(self, x):
        # 根据输入数据形状进行不同处理
        if len(x.shape) == 4:  # 图像数据 (batch, 1, height, width)
            # 移除通道维度，转为 (batch, height, width)
            x = x.squeeze(1)
        elif len(x.shape) == 3:  # TBM数据 (batch, channels, seq_length)
            # 将形状从 (batch, channels, seq_length) 转换为 (batch, seq_length, channels)
            x = x.permute(0, 2, 1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播BiLSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size*2)
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, seq_length, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        # 将输入特征映射到模型维度
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_length)
        
        # Transformer编码器层
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
        self.seq_length = seq_length
        
    def forward(self, x):
        # x shape: [batch_size, channels, seq_length]
        # 转换为Transformer期望的形状 [seq_length, batch_size, features]
        x = x.permute(2, 0, 1)  # [seq_length, batch_size, channels]
        
        # 映射到模型维度
        x = self.embedding(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 使用序列的平均值进行分类
        x = x.mean(dim=0)  # [batch_size, d_model]
        
        # 分类层
        x = self.classifier(x)
        return x


def create_transformer(input_dim=3, seq_length=32, num_classes=10):
    """创建Transformer模型"""
    d_model = 128  # 模型维度
    nhead = 8  # 多头注意力的头数
    num_encoder_layers = 6  # 编码器层数
    dim_feedforward = 512  # 前馈网络维度
    dropout = 0.1  # dropout率
    
    return TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        seq_length=seq_length,
        num_classes=num_classes,
        dropout=dropout
    )
