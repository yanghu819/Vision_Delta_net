
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F


class DeltaNetBlock(nn.Module):
    def __init__(self, hidden_dim, expanded_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expanded_dim = expanded_dim
        
        # 双重归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(expanded_dim)
        
        # 投影层
        self.proj_in = nn.Linear(hidden_dim, expanded_dim * 3)
        self.beta_proj = nn.Linear(hidden_dim, expanded_dim)
        self.proj_out = nn.Linear(expanded_dim, hidden_dim)
        
        # MLP块
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 改进的局部特征处理
        self.conv_layer = nn.Conv1d(
            expanded_dim, expanded_dim,
            kernel_size=3, padding=1,
            groups=expanded_dim, bias=False
        )
        
        # 额外的注意力缩放因子
        self.attention_scale = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        # 使用更好的初始化策略
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.constant_(self.proj_in.bias, 0)
        nn.init.constant_(self.proj_out.bias, 0)
        
        # beta投影层特殊初始化
        nn.init.constant_(self.beta_proj.weight, 0)
        nn.init.constant_(self.beta_proj.bias, 1)
        
        # MLP初始化
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _apply_conv(self, x):
        # 辅助函数：应用卷积层
        return self.conv_layer(x.transpose(1, 2)).transpose(1, 2)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        
        # QKV投影
        qkv = self.proj_in(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        # 激活函数
        q = F.silu(q)
        k = F.silu(k)
        v = F.gelu(v)
        
        # 归一化
        q = F.normalize(q, dim=-1) + 0.1*k
        k = F.normalize(k, dim=-1) + 0.1*q
        
        # 局部特征处理
        q = self._apply_conv(q)
        k = self._apply_conv(k)
        v = self._apply_conv(v)
        
        # 状态更新
        B, L, _ = x.shape
        state = torch.zeros(B, self.expanded_dim, self.expanded_dim, device=x.device)
        
        # 计算beta并限制范围
        beta = torch.sigmoid(self.beta_proj(x)) * 0.9 + 0.1  # [B, L, expanded_dim]
        
        # Delta更新逻辑 - 修正维度
        k_t = k.transpose(-2, -1)  # [B, expanded_dim, L]
        v_old = torch.matmul(state, k_t)  # [B, expanded_dim, L]
        v_new = beta * v + (1 - beta) * v_old.transpose(-2, -1)  # [B, L, expanded_dim]
        
        # 更新state
        state = state + torch.matmul(v_new.transpose(-2, -1), k)  # [B, expanded_dim, expanded_dim]
        
        # 计算输出
        out = torch.matmul(q, state.transpose(-2, -1))
        out = out * self.attention_scale
        out = self.norm2(out)
        out = self.proj_out(out)
        
        # 残差连接和MLP
        x = shortcut + out
        x = x + self.mlp(self.norm1(x))
        
        return x


class VisionDeltaNet(nn.Module):
    def __init__(
        self, 
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        hidden_dim=192,
        expanded_dim=384,
        depth=6
    ):
        super().__init__()
        
        # Patch embedding with overlap
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, stride=2, padding=1),
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        # Improved positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # DeltaNet blocks
        self.blocks = nn.ModuleList([
            DeltaNetBlock(hidden_dim, expanded_dim)
            for _ in range(depth)
        ])
        
        # Layer norm and head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化patch embedding
        for m in self.patch_embed.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
        
        # 初始化分类头
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加分类token和位置编码
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 通过DeltaNet blocks
        for block in self.blocks:
            x = block(x)
        
        # 分类头
        x = self.norm(x)
        x = self.head(x[:, 0])
        
        return x


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
    return val_loss / len(val_loader), 100. * correct / total


def main():
    # 设置随机种子
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    # 加载数据
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # 创建模型
    model = VisionDeltaNet(
        img_size=32,
        patch_size=4,
        num_classes=10,
        hidden_dim=192,
        expanded_dim=384,
        depth=6
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    
    # 学习率调度器
    total_steps = len(trainloader) * 10  # 10 epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 训练循环
    epochs = 10
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, scheduler, device)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')


if __name__ == '__main__':
    main()
