import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
import numpy as np
import time
import os
import sys
import requests
import zipfile
import io

# 尝试导入 psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==========================================
# 0. NVIDIA 4070 Ti 极速配置 (CUDA Mode)
# ==========================================
# 开启 TensorFloat-32 (TF32)，针对 Ampere/Ada 架构(30系/40系)的物理加速
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    device_compute = torch.device("cuda")
    device_structure = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🦖 CUDA Speed Mode Activated: {gpu_name}")
    print(f"   [TF32 Enabled] [AMP Enabled]")
else:
    print("⚠️ 未检测到 GPU，正在使用 CPU (极慢警告)")
    device_compute = torch.device("cpu")
    device_structure = torch.device("cpu")

CONFIG = {
    'dim': 1024,
    'factor_size': 32,
    'initial_rank': 8,
    'max_rank': 64,
    'depth': 8,
    'seq_len': 256,
    
    # ⚡️ 修正后的稳健配置 (针对 16GB 显存) ⚡️
    # 物理批次设为 48，保证中间展开时显存不爆
    'batch_size': 48,           
    
    # 逻辑批次 = 48 * 4 = 192
    # 这样模型学到的效果和 Batch=200 差不多，但显存占用只有 1/4
    'grad_accum_steps': 4,      
    
    'chunk_size': 0,           
    'use_checkpoint': False,    # 如果依然 OOM，把这里改成 True
    
    'lr': 1.2e-3,               # 恢复稍微高一点的学习率
    'weight_decay': 0.02,
    'dropout_rate': 0.1,
    'growth_interval': 1000,
    'eval_interval': 50,       
    'axiom_lambda': 0.05,
    'total_iters': 100000,
    
    'checkpoint_path': 'qrlm_cuda.pt',
    'best_model_path': 'qrlm_best_cuda.pt',
    'data_dir': 'data_wikitext',
}

# ==========================================
# 1. 监控工具
# ==========================================
def format_log(step, train_loss, val_loss, grad_norm, rank, energy, dt):
    diff = train_loss - val_loss
    # 获取显存使用量 (GB)
    vram = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    
    c_reset = "\033[0m"
    c_diff = "\033[92m" if diff < 0 else "\033[91m"
    c_mem = "\033[96m"
    
    return (f"Step {step:6d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Diff: {c_diff}{diff:+.4f}{c_reset} | Grad: {grad_norm:.2f} | "
            f"Rank: {rank:2d} | Energy: {energy:.2f} | "
            f"VRAM: {c_mem}{vram:.2f}GB{c_reset} | "
            f"Time: {dt:.1f}ms")

# ==========================================
# 2. 数据集加载器
# ==========================================
class WikiTextLoader:
    def __init__(self, block_size, batch_size, data_dir):
        self.block_size = block_size
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'wiki.train.raw')
        self.val_path = os.path.join(data_dir, 'wiki.valid.raw')
        self._prepare_data()
        print("📚 读取 WikiText 数据...")
        with open(self.train_path, 'r', encoding='utf-8') as f: train_text = f.read()
        with open(self.val_path, 'r', encoding='utf-8') as f: val_text = f.read()
        text = train_text + val_text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.train_data = torch.tensor([self.stoi[c] for c in train_text], dtype=torch.long)
        self.val_data = torch.tensor([self.stoi[c] for c in val_text], dtype=torch.long)

    def _prepare_data(self):
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        if os.path.exists(self.train_path): return
        print("📦 下载 WikiText-2...")
        url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip'
        try:
            r = requests.get(url, stream=True, timeout=30)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.data_dir)
            base = os.path.join(self.data_dir, 'wikitext-2-raw')
            os.rename(os.path.join(base, 'wiki.train.raw'), self.train_path)
            os.rename(os.path.join(base, 'wiki.valid.raw'), self.val_path)
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            sys.exit(1)

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device_compute), y.to(device_compute)
    
    def decode(self, l): return ''.join([self.itos[i] for i in l])

# ==========================================
# 3. 核心组件 (针对 CUDA 优化)
# ==========================================

class WaveStructureBank(nn.Module):
    def __init__(self, num_blocks, max_rank):
        super().__init__()
        self.num_blocks = num_blocks 
        self.sub_blocks = num_blocks // 4 
        self.max_rank = max_rank
        self.factors_A_components = nn.ParameterList() 
        self.current_rank = 0
        
    def request_rank(self, target_rank):
        while self.current_rank < target_rank:
            # CUDA 完整支持 QR 分解，直接在 GPU 初始化
            comps = torch.randn(4, self.sub_blocks, self.sub_blocks, device=device_structure, dtype=torch.float32)
            for c in range(4):
                nn.init.orthogonal_(comps[c])
            scale = (self.current_rank + 1) ** -0.5
            self.factors_A_components.append(nn.Parameter(comps * scale))
            self.current_rank += 1
            
    def get_factors(self, rank):
        return self.factors_A_components[:rank]

class BalancedHamiltonLayer(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, initial_rank=1):
        super().__init__()
        self.dim = dim
        self.factor_size = factor_size
        self.num_blocks = dim // factor_size
        self.sub_blocks = self.num_blocks // 4
        self.structure_bank = structure_bank
        
        self.factors_B_gpu = nn.ParameterList()
        self.current_rank = 0
        self.bias = nn.Parameter(torch.zeros(dim, device=device_compute))
        
        for _ in range(initial_rank):
            self.add_rank()

    def add_rank(self):
        self.structure_bank.request_rank(self.current_rank + 1)
        b = torch.randn(self.factor_size, self.factor_size, device=device_compute)
        nn.init.orthogonal_(b)
        scale = (self.current_rank + 1) ** -0.5
        self.factors_B_gpu.append(nn.Parameter(b * scale))
        self.current_rank += 1

    def _construct_hamilton_matrix_batch(self, A_comps_stack):
        r = A_comps_stack[:, 0]
        i = A_comps_stack[:, 1]
        j = A_comps_stack[:, 2]
        k = A_comps_stack[:, 3]
        
        row0 = torch.cat([r, -i, -j, -k], dim=2)
        row1 = torch.cat([i, r, -k, j], dim=2)
        row2 = torch.cat([j, k, r, -i], dim=2)
        row3 = torch.cat([k, -j, i, r], dim=2)
        H = torch.cat([row0, row1, row2, row3], dim=1)
        return H

    def _forward_impl(self, x):
        # ⚡️ 4070 Ti 全量计算模式 (已修复维度 bug) ⚡️
        B_batch, T, D = x.shape
        
        # 1. 展平数据
        # 维度: [Total_Tokens (n), 4*Sub (s), Factor (i)]
        # 之前报错是因为 einsum 漏掉了中间的 's' 维度
        x_flat = x.view(-1, 4 * self.sub_blocks, self.factor_size)
        
        # 2. 准备权重
        factors_A_list = self.structure_bank.get_factors(self.current_rank)
        A_stack = torch.stack([p.to(dtype=x.dtype) for p in factors_A_list], dim=0)
        B_stack = torch.stack([p.to(dtype=x.dtype) for p in self.factors_B_gpu], dim=0)
        
        H_stack = self._construct_hamilton_matrix_batch(A_stack)
        
        # 3. 核心全量计算
        
        # Step 1: Local Mixing (因子混合)
        # x_flat: [n, s, i] (Total, SubBlocks, FactorIn)
        # B_stack: [r, j, i] (Rank, FactorOut, FactorIn)
        # Output: [r, n, s, j]
        # 修正点：'nsi, rji -> rnsj' (正确处理了3维输入)
        wave_mod = torch.einsum('nsi, rji -> rnsj', x_flat, B_stack)
        
        # Step 2: Global Mixing (哈密尔顿混合) & Rank Aggregation
        # wave_mod: [r, n, s, j]
        # H_stack:  [r, k, s] (Rank, SubOut, SubIn)
        # Output:   [n, k, j]
        wave_out = torch.einsum('rnsj, rks -> nkj', wave_mod, H_stack)
            
        return wave_out.reshape(B_batch, T, D) + self.bias

    def forward(self, x):
        if self.training and CONFIG['use_checkpoint']:
            return checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def orthogonality_loss(self):
        loss = torch.tensor(0.0, device=device_compute)
        for p in self.factors_B_gpu:
            p_f32 = p.float()
            gram = torch.mm(p_f32.t(), p_f32)
            eye = torch.eye(p.shape[1], device=device_compute)
            loss = loss + torch.norm(gram - eye)
        return loss

class BalancedFFN(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, initial_rank):
        super().__init__()
        self.fc1 = BalancedHamiltonLayer(dim, factor_size, structure_bank, initial_rank)
        self.act = nn.GELU()
        self.fc2 = BalancedHamiltonLayer(dim, factor_size, structure_bank, initial_rank)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))
    def grow(self): self.fc1.add_rank(); self.fc2.add_rank()

class BalancedAttention(nn.Module):
    def __init__(self, dim, factor_size, structure_bank, initial_rank, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, initial_rank)
        self.k_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, initial_rank)
        self.v_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, initial_rank)
        self.o_proj = BalancedHamiltonLayer(dim, factor_size, structure_bank, initial_rank)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=device_compute) * float('-inf'), diagonal=1)
        att = F.softmax(att + mask, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class QRLM_Balanced(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        num_blocks = config['dim'] // config['factor_size']
        self.structure_bank = WaveStructureBank(num_blocks, config['max_rank'])
        self.token_emb = nn.Embedding(vocab_size, config['dim'])
        self.pos_emb = nn.Parameter(torch.randn(1, config['seq_len'], config['dim']) * 0.02)
        self.drop = nn.Dropout(config['dropout_rate'])
        self.layers = nn.ModuleList()
        for _ in range(config['depth']):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(config['dim']),
                'attn': BalancedAttention(config['dim'], config['factor_size'], self.structure_bank, config['initial_rank']),
                'norm2': nn.LayerNorm(config['dim']),
                'ffn': BalancedFFN(config['dim'], config['factor_size'], self.structure_bank, config['initial_rank'])
            }))
        self.head = nn.Linear(config['dim'], vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.shape
        x = self.drop(self.token_emb(x) + self.pos_emb[:, :T, :])
        ortho_loss = torch.tensor(0.0, device=device_compute)
        wave_energy = x.norm(dim=-1).mean()
        for layer in self.layers:
            x = x + layer['attn'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))
            ortho_loss = ortho_loss + layer['attn'].q_proj.orthogonality_loss()
            ortho_loss = ortho_loss + layer['ffn'].fc1.orthogonality_loss()
        logits = self.head(x)
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + self.config['axiom_lambda'] * ortho_loss * 0.01
        return logits, loss, wave_energy

    def grow_network(self):
        print(f"\n🌱 [Growth] Rank {self.layers[0]['ffn'].fc1.current_rank} -> {self.layers[0]['ffn'].fc1.current_rank + 1}")
        for layer in self.layers: layer['ffn'].grow()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['seq_len']:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 5. 训练循环 (引入 AMP)
# ==========================================
@torch.no_grad()
def estimate_loss(model, loader, eval_iters=10):
    model.eval()
    losses = torch.zeros(eval_iters, device=device_compute)
    for k in range(eval_iters):
        X, Y = loader.get_batch('val')
        # Eval 时不需要 scaler，但可以用 autocast 加速
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            _, loss, _ = model(X, Y)
        losses[k] = loss
    model.train()
    return losses.mean()

def train_balanced_system():
    loader = WikiTextLoader(CONFIG['seq_len'], CONFIG['batch_size'], CONFIG['data_dir'])
    model = QRLM_Balanced(loader.vocab_size, CONFIG)
    model.to(device_compute)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 物理参数: {total_params/1e6:.2f}M")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # ⚡️ 初始化混合精度 Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    start_iter = 0
    loss_history = []
    best_val_loss = float('inf')
    
    if os.path.exists(CONFIG['checkpoint_path']):
        print(f"🔄 恢复存档: {CONFIG['checkpoint_path']}")
        ckpt = torch.load(CONFIG['checkpoint_path'], map_location=device_compute, weights_only=False)
        saved_rank = ckpt['current_rank']
        while model.layers[0]['ffn'].fc1.current_rank < saved_rank:
            model.grow_network()
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # 恢复 scaler 状态
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
            
        start_iter = ckpt['iter']
        loss_history = ckpt.get('loss_history', [])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✅ 恢复至 Step {start_iter}, Best Val: {best_val_loss:.4f}")

    print("🚀 开始 CUDA 极速训练...")
    
    try:
        model.train()
        optimizer.zero_grad()
        
        for iter_num in range(start_iter, CONFIG['total_iters']):
            t0 = time.time()
            accum_loss = 0
            accum_energy = 0
            
            for _ in range(CONFIG['grad_accum_steps']):
                xb, yb = loader.get_batch('train')
                
                # ⚡️ 混合精度前向传播
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits, loss, energy = model(xb, yb)
                    loss = loss / CONFIG['grad_accum_steps']
                
                # ⚡️ Scaler 反向传播
                scaler.scale(loss).backward()
                
                accum_loss += loss.item()
                accum_energy += energy.item()
            
            accum_energy /= CONFIG['grad_accum_steps']
            
            # ⚡️ Scaler 更新参数
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_history.append(accum_loss)
            dt = (time.time() - t0) * 1000
            
            # 生长策略
            if iter_num > 2000 and iter_num % CONFIG['growth_interval'] == 0:
                recent = np.mean(loss_history[-100:])
                old = np.mean(loss_history[-2000:-1900])
                if recent > old * 0.99 and model.layers[0]['ffn'].fc1.current_rank < CONFIG['max_rank']:
                    print(f"\n⚠️ Loss 停滞 ({old:.3f}->{recent:.3f})，触发生长...")
                    model.grow_network()
                    for pg in optimizer.param_groups: pg['lr'] *= 0.8

            if iter_num % CONFIG['eval_interval'] == 0:
                val_loss = estimate_loss(model, loader)
                current_rank = model.layers[0]['ffn'].fc1.current_rank
                print(format_log(iter_num, accum_loss, val_loss, total_norm, current_rank, accum_energy, dt))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = {
                        'iter': iter_num, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'current_rank': current_rank, 'loss_history': loss_history, 'best_val_loss': best_val_loss, 'config': CONFIG
                    }
                    torch.save(best_ckpt, CONFIG['best_model_path'])
                    # print(f"🌟 Saved Best: {best_val_loss:.4f}")

            if iter_num > 0 and iter_num % 1000 == 0:
                ckpt = {
                    'iter': iter_num, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'current_rank': model.layers[0]['ffn'].fc1.current_rank, 'loss_history': loss_history, 'best_val_loss': best_val_loss
                }
                torch.save(ckpt, CONFIG['checkpoint_path'])
                print("\n📜 [Generation]:")
                model.eval()
                with torch.no_grad():
                    ctx = torch.zeros((1, 1), dtype=torch.long, device=device_compute)
                    out = model.generate(ctx, 100)
                    print(loader.decode(out[0].tolist()))
                model.train()
                print("-" * 50)

    except KeyboardInterrupt:
        print("\n🛑 保存并退出...")
        ckpt = {
            'iter': iter_num, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'current_rank': model.layers[0]['ffn'].fc1.current_rank, 'loss_history': loss_history, 'best_val_loss': best_val_loss
        }
        torch.save(ckpt, CONFIG['checkpoint_path'])

if __name__ == "__main__":
    train_balanced_system()