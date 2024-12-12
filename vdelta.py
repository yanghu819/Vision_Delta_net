import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# 简化版 Muon 优化器实现
@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    使用 Newton-Schulz 迭代计算矩阵的零次方/正交化。
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class SimpleMuon(torch.optim.Optimizer):
    """
    简化版 Muon 优化器 - 去除分布式相关代码
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                g = p.grad
                state = self.state[p]
                
                # 初始化动量缓冲区
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                    
                buf = state['momentum_buffer']
                buf.lerp_(g, 1 - momentum)
                
                # 使用 Nesterov 动量
                if nesterov:
                    g = g.lerp(buf, momentum)
                else:
                    g = buf
                    
                # 应用 Newton-Schulz 迭代
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                
                # 更新参数
                alpha = -lr * max(1, p.size(0) / p.size(1)) ** 0.5
                p.data.add_(g.view_as(p), alpha=alpha)


class DeltaNetBlock(nn.Module):
    def __init__(self, hidden_dim, expanded_dim, max_seq_length=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expanded_dim = expanded_dim
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(expanded_dim)
        
        # 投影层
        self.proj_in = nn.Linear(hidden_dim, expanded_dim * 3)
        self.proj_out = nn.Linear(expanded_dim, hidden_dim)
        
        # Position-aware beta 相关层
        self.rel_pos_embedding = nn.Parameter(torch.zeros(2 * max_seq_length - 1, hidden_dim))
        self.beta_pos_proj = nn.Linear(hidden_dim * 2, expanded_dim)
        
        # MLP块
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 局部特征处理
        self.conv_layer = nn.Conv1d(
            expanded_dim, expanded_dim,
            kernel_size=3, padding=1,
            groups=expanded_dim, bias=False
        )
        
        # 注意力缩放
        self.attention_scale = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        # 初始化投影层
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.constant_(self.proj_in.bias, 0)
        nn.init.constant_(self.proj_out.bias, 0)
        
        # 初始化相对位置编码
        nn.init.normal_(self.rel_pos_embedding, std=0.02)
        
        # 初始化MLP
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_rel_pos_emb(self, length):
        """获取相对位置编码"""
        center = length - 1
        pos_ids = torch.arange(length, device=self.rel_pos_embedding.device)
        rel_pos_ids = pos_ids.unsqueeze(0) - pos_ids.unsqueeze(1) + center
        return self.rel_pos_embedding[rel_pos_ids]
    
    def _apply_conv(self, x):
        return self.conv_layer(x.transpose(1, 2)).transpose(1, 2)
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        
        # 投影得到q,k,v
        qkv = self.proj_in(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        # 非线性激活
        q = F.silu(q)
        k = F.silu(k)
        v = F.gelu(v)
        
        # 范数和残差连接
        q = F.normalize(q, dim=-1) + 0.1 * k
        k = F.normalize(k, dim=-1) + 0.1 * q
        
        # 卷积处理
        q = self._apply_conv(q)
        k = self._apply_conv(k)
        v = self._apply_conv(v)
        
        # 获取批次大小和序列长度
        B, L, _ = x.shape
        
        # 修改这部分代码来处理位置信息
        rel_pos = self.get_rel_pos_emb(L)  # [L, L, hidden_dim]
        
        # 计算位置感知信息 - 修正维度处理
        x_flat = x.view(-1, self.hidden_dim)  # [B*L, hidden_dim]
        pos_embed = self.rel_pos_embedding[:L]  # [L, hidden_dim]
        pos_info = torch.matmul(x_flat, pos_embed.T)  # [B*L, L]
        pos_info = pos_info.view(B, L, L)  # [B, L, L]
        pos_info = torch.mean(pos_info, dim=-1, keepdim=True)  # [B, L, 1]
        pos_info = pos_info.expand(-1, -1, self.hidden_dim)  # [B, L, hidden_dim]
        
        # 结合内容和位置信息计算beta
        combined_features = torch.cat([x, pos_info], dim=-1)
        beta = torch.sigmoid(self.beta_pos_proj(combined_features)) * 0.9 + 0.1
        
        # 状态更新
        state = torch.zeros(B, self.expanded_dim, self.expanded_dim, device=x.device)
        k_t = k.transpose(-2, -1)
        v_old = torch.matmul(state, k_t)
        v_new = beta * v + (1 - beta) * v_old.transpose(-2, -1)
        state = state + torch.matmul(v_new.transpose(-2, -1), k)
        
        # 输出计算
        out = torch.matmul(q, state.transpose(-2, -1))
        out = out * self.attention_scale
        out = self.norm2(out)
        out = self.proj_out(out)
        
        # 残差连接和MLP
        x = shortcut + out
        x = x + self.mlp(self.norm1(x))
        
        return x

class VisionDeltaNet(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 hidden_dim=384, expanded_dim=768, depth=12):
        super().__init__()
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, stride=2, padding=1),
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        self.blocks = nn.ModuleList([
            DeltaNetBlock(hidden_dim, expanded_dim)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.patch_embed.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

def train_epoch(model, train_loader, criterion, optimizers, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        # 清零所有优化器的梯度
        for opt in optimizers:
            opt.zero_grad()
            
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新所有优化器
        for opt in optimizers:
            opt.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    
    model = VisionDeltaNet(
        img_size=32, 
        patch_size=4, 
        num_classes=10, 
        hidden_dim=384, 
        expanded_dim=768, 
        depth=12
    ).to(device)
    
    # 分离不同类型的参数
    matrix_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.ndim == 2 and 'weight' in name and not any(x in name for x in ['embed', 'head']):
            matrix_params.append(param)
        else:
            other_params.append(param)
    
    # 创建优化器
    optimizers = [
        SimpleMuon(matrix_params, lr=0.02, momentum=0.95, nesterov=True),
        optim.AdamW(other_params, lr=0.001, weight_decay=0.05)
    ]
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    epochs = 300
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizers, device)
        val_loss, val_acc = validate(model, testloader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

if __name__ == '__main__':
    main()
