import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from efficientnet_pytorch.model import MemoryEfficientSwish
from typing import List
import numbers


from thop import profile, clever_format
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Efficient Frequency-Domain Attention
class EFA(nn.Module):
    def __init__(self, dim, bias=False):
        super(EFA, self).__init__()
        # 定义一个卷积层，用于将输入从dim维度转换到dim * 6维度
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)

        # 定义一个深度可分离卷积层，保持输出维度为dim * 6
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        # 定义一个投影层，用于将dim * 2维度的数据映射回dim维度
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        # 应用LayerNorm归一化，指定类型为'WithBias'
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        # 设置patch大小为8
        self.patch_size = 8

    def forward(self, x):
        # input_tensor = torch.randn(1, 64, 32, 32)
        # 将输入x通过to_hidden层得到hidden特征图
        hidden = self.to_hidden(x) # torch.Size([1, 384, 32, 32])

        # 对hidden进行深度可分离卷积后分割成q、k、v三个部分
        # Q torch.Size([1, 128, 32, 32])
        # K torch.Size([1, 128, 32, 32])
        # V torch.Size([1, 128, 32, 32])
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # 重新排列q k 以适应patch尺寸
        # Q P = torch.Size([1, 128, 4, 4, 8, 8])
        # K P = torch.Size([1, 128, 4, 4, 8, 8])
        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)

        # 对q k 执行二维实数快速傅里叶变换，并且在在频域上计算q和k的乘积
        """
                torch.fft.rfft2 函数执行的是二维实数快速傅里叶变换（Real-to-Complex 2D FFT）。
                当你对一个实数值的张量进行这种变换时，输出的张量在最后一个维度上会减少到一半，然后加上1。
                这是因为对于实数输入，其傅里叶变换的结果是共轭对称的，所以只需要保存一半的数据外加零频点就可以完全恢复整个频谱.
                因此，最后一个维度从 8 减少到了 (8 // 2) + 1 = 5
        """
        q_fft = torch.fft.rfft2(q_patch.float()) # torch.Size([1, 128, 4, 4, 8, 5])
        k_fft = torch.fft.rfft2(k_patch.float()) # torch.Size([1, 128, 4, 4, 8, 5])
        out = q_fft * k_fft # torch.Size([1, 128, 4, 4, 8, 5])

        # 逆向傅里叶变换回到空间域
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size)) # torch.Size([1, 128, 4, 4, 8, 8])
        # 重新排列out回到原始形状
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size) # torch.Size([1, 128, 32, 32])
        # 对结果应用LayerNorm归一化
        out = self.norm(out)

        # 计算最终输出，即v与out的元素级相乘
        # V torch.Size([1, 128, 32, 32])
        # Out torch.Size([1, 128, 32, 32])
        output = v * out

        # 通过project_out层调整维度
        # torch.Size([1, 64, 32, 32])
        output = self.project_out(output)

        return output
class AttnMap(nn.Module):
    """
    注意力映射模块（用于高频注意力分支的特征交互）
    功能：通过双1×1卷积+Swish激活，对QK乘积后的特征进行非线性变换，增强注意力表达能力
    """
    def __init__(self, dim):
        """
        Args:
            dim: 输入/输出特征通道数（需与高频分支的头维度匹配）
        """
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),  # 1×1卷积：通道内特征变换
            MemoryEfficientSwish(),        # 高效Swish激活：引入非线性，避免梯度消失
            nn.Conv2d(dim, dim, 1, 1, 0)   # 1×1卷积：进一步优化特征分布
            # nn.Identity()  # 备用：无变换模式，调试时可启用
        )

    def forward(self, x):
        """前向传播：输入特征 → 双1×1卷积+激活 → 输出"""
        # (b, dim, h, w) → (b, dim, h, w)，经过AttnMap不改变特征维度
        return self.act_block(x)


class EfficientAttention(nn.Module):
    """
    高效注意力模块（EfficientAttention）
    设计思想：分“高频注意力分支”与“低频注意力分支”并行处理特征，
              高频分支捕捉局部细节，低频分支捕捉全局上下文，平衡精度与效率

    特征维度需要被注意力头数整除；
    有几个头分组列表的长度就是几；
    卷积核列表的长度是头数减一；
    根据自己实际需要设置上述参数；
    """

    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=8,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        """
        Args:
            dim: 输入特征通道数（需被num_heads整除）
            num_heads: 注意力头总数（高频分支头数之和 + 低频分支头数 = num_heads）
            group_split: 各分支的注意力头分配列表（最后一个元素为低频分支头数，其余为高频分支）
            kernel_sizes: 高频分支的深度卷积核大小列表（数量 = 高频分支数）
            window_size: 低频分支的平均池化窗口大小（默认7，用于压缩特征尺寸，降低计算量）
            attn_drop: 注意力 dropout 概率（默认0，正则化，防止过拟合）
            proj_drop: 输出投影 dropout 概率（默认0，正则化）
            qkv_bias: QKV生成卷积是否使用偏置（默认True，提升模型表达能力）
        """
        super().__init__()
        # 校验参数合法性：头分配总和=总头数，高频分支数=卷积核列表长度
        assert sum(group_split) == num_heads, "Sum of group_split must equal num_heads"
        assert len(kernel_sizes) + 1 == len(group_split), "Length of kernel_sizes +1 must equal length of group_split"

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads  # 单个注意力头的通道数
        self.scalor = self.dim_head ** -0.5# 注意力缩放因子（避免QK乘积过大）
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split

        # 初始化高频分支组件：深度卷积（mixer）、注意力映射（act_blocks）、QKV生成卷积（qkvs）
        convs = []  # 高频分支：深度卷积（分组数=3×头通道数×分支头数，实现通道独立变换）
        act_blocks = []  # 高频分支：AttnMap模块（增强QK交互）
        qkvs = []  # 高频分支：QKV生成（1×1卷积，输出通道=3×分支头数×头通道数）

        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]  # 当前高频分支的头数
            if group_head == 0:
                continue# 头数为0时跳过（兼容灵活配置）

            # 深度卷积：输入输出通道=3×分支头数×头通道数，分组卷积确保各QKV通道独立
            convs.append(nn.Conv2d(
                in_channels=3 * self.dim_head * group_head,
                out_channels=3 * self.dim_head * group_head,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,  # same padding：输入输出尺寸一致
                groups=3 * self.dim_head * group_head  # 深度卷积：每组对应一个Q/K/V通道
            ))

            # 注意力映射模块：输入通道=分支头数×头通道数
            act_blocks.append(AttnMap(self.dim_head * group_head))

            # QKV生成卷积：将输入dim通道转换为3×分支头数×头通道数（Q/K/V各占1/3）， (b, dim, h, w) → (b, 3×group_head×dim_head, h, w)
            qkvs.append(nn.Conv2d(
                in_channels=dim,
                out_channels=3 * group_head * self.dim_head,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=qkv_bias
            ))

        # 初始化低频分支组件（全局注意力）：仅当头数分配非0时生效
        if group_split[-1] != 0:
            # 全局Q生成：(b, dim, h, w) → (b, group_split[-1]×dim_head, h, w)
            self.global_q = nn.Conv2d(  # 低频分支Q生成：输出通道=低频头数×头通道数
                dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias
            )
            # 全局KV生成：(b, dim, h, w) → (b, 2×group_split[-1]×dim_head, h, w)
            self.global_kv = nn.Conv2d(  # 低频分支KV生成：输出通道=2×低频头数×头通道数（K/V各占1/2）
                dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias
            )
            # 平均池化：压缩特征尺寸（步长=窗口大小），降低低频分支计算量
            # (b, c, h, w) → (b, c, h//window_size, w//window_size)（步长=窗口大小）
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1else nn.Identity()

        # 注册模块列表（确保PyTorch可追踪参数）
        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)

        # 输出投影：将多分支拼接后的特征映射回原dim通道
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        # Dropout层：正则化
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        """
        高频注意力分支前向传播（捕捉局部细节特征，如边缘、纹理）
        Args:
            x: 输入特征，shape=(b, c, h, w)
            to_qkv: QKV生成卷积模块
            mixer: 深度卷积模块（特征混合）
            attn_block: AttnMap模块（注意力映射）
        Returns:
            res: 高频注意力输出，shape=(b, group_head×dim_head, h, w)
        """
        b, c, h, w = x.size()
        # 1. 生成QKV：输入dim → 3×group_head×dim_head
        qkv = to_qkv(x)  # shape=(b, 3×m×d, h, w)，m=分支头数，d=头通道数
        # 2. 深度卷积混合：分组卷积增强QKV局部交互，后重塑为(3, b, m×d, h, w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()
        # 3. 拆分Q、K、V：各shape=(b, m×d, h, w)
        q, k, v = qkv
        # 4. 注意力计算：Q×K（元素乘）→ AttnMap映射 → 缩放 → Tanh激活 → Dropout
        attn = attn_block(q.mul(k)).mul(self.scalor)  # QK元素乘后非线性变换，再缩放
        attn = self.attn_drop(torch.tanh(attn))  # Tanh激活限制注意力范围，Dropout正则化
        # 5. 注意力加权V：Attn × V（元素乘）
        res = attn.mul(v)  # shape=(b, m×d, h, w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        """
        低频注意力分支前向传播（捕捉全局上下文特征，如目标整体结构）
        Args:
            x: 输入特征，shape=(b, c, h, w)
            to_q: Q生成卷积模块
            to_kv: KV生成卷积模块
            avgpool: 平均池化模块（压缩KV尺寸）
        Returns:
            res: 低频注意力输出，shape=(b, group_head×dim_head, h, w)
        """
        b, c, h, w = x.size()
        # 1. 生成Q：输入dim → group_head×dim_head，重塑为多头注意力格式
        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
        # shape=(b, m, h×w, d)，m=低频分支头数，d=头通道数

        # 2. 生成KV：先池化压缩尺寸（降低计算量），再生成KV
        kv = avgpool(x)  # shape=(b, c, H, W)，H=h//window_size，W=w//window_size
        kv = to_kv(kv).view(  # 生成KV并重塑为多头格式
            b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)
        ).permute(1, 0, 2, 4, 3).contiguous()
        # shape=(2, b, m, H×W, d)，拆分后K/V shape=(b, m, H×W, d)

        # 3. 拆分K、V
        k, v = kv
        # 4. 注意力计算：Q@K^T（矩阵乘）→ 缩放 → Softmax → Dropout
        attn = self.scalor * q @ k.transpose(-1, -2)  # shape=(b, m, h×w, H×W)
        attn = self.attn_drop(attn.softmax(dim=-1))  # Softmax归一化注意力权重
        # 5. 注意力加权V：Attn@V（矩阵乘），后重塑为空间特征格式
        res = attn @ v  # shape=(b, m, h×w, d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()  # shape=(b, m×d, h, w)
        return res

    def forward(self, x: torch.Tensor):
        """
        整体前向传播：高频分支并行计算 → 低频分支计算 → 分支拼接 → 输出投影
        Args:
            x: 输入特征，shape=(b, c, h, w)
        Returns:
            最终注意力输出，shape=(b, dim, h, w)（与输入尺寸一致）
        """
        res = []  # 存储各分支输出
        # 1. 高频分支计算：遍历所有高频分支，添加输出到res
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue# 跳过头数为0的分支
            res.append(self.high_fre_attntion(
                x, self.qkvs[i], self.convs[i], self.act_blocks[i]
            ))
        # 2. 低频分支计算：若头数非0，添加输出到res
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(
                x, self.global_q, self.global_kv, self.avgpool
            ))
        # 3. 分支拼接 → 输出投影 → Dropout：恢复原dim通道，返回最终结果
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


class EDFFN(nn.Module):
    def __init__(self, dim, patch_size, ffn_expansion_factor=4, bias=True):
        super(EDFFN, self).__init__()
        # 计算隐藏层的特征维度，通常是输入维度的若干倍
        hidden_features = int(dim * ffn_expansion_factor)
        # 保存patch大小，用于后续分块处理
        self.patch_size = patch_size
        self.dim = dim
        # 第一个1x1卷积层，用于提升特征维度，输出维度是隐藏层维度的两倍
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # 深度可分离卷积，对每个通道单独处理，进一步提取特征
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # 可学习的FFT参数，用于频域操作
        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        # 第二个1x1卷积层，用于将特征维度降回输入维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    #@torch.compiler.disable
    def forward(self, x):
        # 通过第一个卷积层提升特征维度【提升维度】
        x = self.project_in(x)
        # 经过深度可分离卷积后，将输出分成两部分
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 对第一部分应用GELU激活函数，然后与第二部分相乘
        x = F.gelu(x1) * x2
        # 通过第二个卷积层降低特征维度【降低维度】
        x = self.project_out(x)

        # 将特征图按指定patch大小进行分块重组
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,patch2=self.patch_size)
        # 对分块后的特征图进行二维快速傅里叶变换，转换到频域
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        # 在频域中应用可学习的参数，对频域特征进行调整
        x_patch_fft = x_patch_fft * self.fft
        # 进行二维逆快速傅里叶变换，将特征从频域转回空间域
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))

        # 将分块的特征图重新组合成完整的特征图
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,patch2=self.patch_size)
        return x


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x


# Efficient Spatial-Domain Attention
class ESA(nn.Module):
    def __init__(self, dim=36):
        super(ESA, self).__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)

        self.lde = DMlp(dim, 2)

        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, f):
        _, _, h, w = f.shape
        y, x = self.linear_0(f).chunk(2, dim=1)
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        y_d = self.lde(y)
        return self.linear_2(x_l + y_d)


class SFFBlock(nn.Module):
    def __init__(self, dim, drop_path=0.,freAtten=False):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        if not freAtten:
            self.mixer = ESA(dim)
        else:
            self.mixer = EFA(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.ffn = EDFFN(dim,patch_size=8)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x): return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x): return self.pixel_shuffle(self.conv(x))


class Conv(nn.Module):
    """标准的卷积模块，包含卷积、归一化和激活函数。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2  # 计算填充大小
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU() if activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class SPP(nn.Module):
    """空间金字塔池化（SPP）层的实现。"""

    def __init__(self, in_channels):
        super(SPP, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1)

        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.conv2 = Conv(hidden_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2(x)
        return x


class ELAN(nn.Module):
    """ELAN 结构的实现。"""

    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()
        hidden_channels = out_channels // 2

        self.branch1_conv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.branch1_conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3)

        self.branch2_conv1 = Conv(in_channels, hidden_channels, kernel_size=1)
        self.branch2_conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3)

        self.concat_conv = Conv(hidden_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1_conv1(x)
        b1 = self.branch1_conv2(b1)

        b2 = self.branch2_conv1(x)
        b2 = self.branch2_conv2(b2)

        concat = torch.cat([b1, b2], dim=1)
        out = self.concat_conv(concat)
        return out


class sppELAN(nn.Module):
    """sppELAN 模块的完整实现。"""

    def __init__(self, in_channels, out_channels):
        super(sppELAN, self).__init__()
        self.initial_conv = Conv(in_channels, in_channels, kernel_size=1)

        self.spp = SPP(in_channels)

        self.elan = ELAN(in_channels, out_channels)

    def forward(self, x):
        x1 = self.initial_conv(x)
        x2 = self.spp(x1)
        x3 = self.elan(x2) + x2
        return x3


class SFFFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=16):
        super().__init__()

        self.stem = nn.Conv2d(in_channels, dim, 3, 1, 1)

        # Stage 0
        self.encode_0 = nn.Sequential(SFFBlock(dim))
        self.down_0 = Downsampling(dim, dim * 2)

        # Stage 1
        self.encode_1 = nn.Sequential(SFFBlock(dim * 2))
        self.down_1 = Downsampling(dim * 2, dim * 4)

        # Stage 2
        self.encode_2 = nn.Sequential(SFFBlock(dim * 4))
        self.down_2 = Downsampling(dim * 4, dim * 8)

        # Bottleneck (8dim)
        self.bottle = nn.Sequential(
            sppELAN(in_channels=dim * 8, out_channels=dim * 8),
            EfficientAttention(dim* 8, 4, [1, 1, 1, 1], [9, 7, 5], window_size=8)
        )
        # Decoder
        self.up_0 = UpsampleBlock(dim * 8, dim * 4)
        self.decode_0 = nn.Sequential(SFFBlock(dim * 8,freAtten=True))

        self.up_1 = UpsampleBlock(dim * 8, dim * 2)
        self.decode_1 = nn.Sequential(SFFBlock(dim * 4,freAtten=True))

        self.up_2 = UpsampleBlock(dim * 4, dim)
        self.decode_2 = nn.Sequential(SFFBlock(dim * 2,freAtten=True))

        self.out = nn.Conv2d(dim * 2, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.stem(x)

        e0 = self.encode_0(x)
        d0 = self.down_0(e0)

        e1 = self.encode_1(d0)
        d1 = self.down_1(e1)

        e2 = self.encode_2(d1)
        d2 = self.down_2(e2)

        b = self.bottle(d2)

        u0 = self.up_0(b)
        dec0 = self.decode_0(torch.cat([u0, e2], dim=1))

        u1 = self.up_1(dec0)
        dec1 = self.decode_1(torch.cat([u1, e1], dim=1))

        u2 = self.up_2(dec1)
        dec2 = self.decode_2(torch.cat([u2, e0], dim=1))

        return self.out(dec2)






def benchmark_steg_lite(model, input_res=(1, 3, 1024, 1024)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化模型
    model = model.to(device)
    input_data = torch.randn(*input_res).to(device)

    # 2. 测试参数量和 FLOPs (使用 thop)
    # 注意：FFT 算子在 thop 中可能被视为 0 FLOPs，取决于版本
    flops, params = profile(model, inputs=(input_data,), verbose=False)
    readable_flops, readable_params = clever_format([flops, params], "%.3f")

    # 3. 测试显存占用
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 推理模式显存
    model.eval()
    with torch.no_grad():
        _ = model(input_data)
    mem_eval = torch.cuda.max_memory_allocated() / 1024 ** 2


    # 4. 打印报告
    print("-" * 30)
    print(f"输入尺寸: {input_res[2]}x{input_res[3]}")
    print(f"模型参数量 (Params): {readable_params}")
    print(f"计算量 (FLOPs):    {readable_flops}")
    print(f"推理显存 (Eval):   {mem_eval:.2f} MB")
    print("-" * 30)
    return params,flops,mem_eval


if __name__ == "__main__":
    encoder = SFFFormer(in_channels=6, out_channels=3, dim=16).cuda()
    decoder = SFFFormer(in_channels=3, out_channels=3, dim=16).cuda()
    a1,b1,c1=benchmark_steg_lite(encoder, input_res=(1, 6, 256, 256))
    a2,b2,c2=benchmark_steg_lite(decoder, input_res=(1, 3, 256, 256))
    params,flops = clever_format([a1+a2, b1+b2], "%.3f")
    print(f"模型参数量 (Params): {params}")
    print(f"计算量 (FLOPs):    {flops}")
    print(f"推理显存 (Eval):   {c1+c2:.2f} MB")

