import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F


# Replace PyTorch's to_2tuple function
def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


# Replace PyTorch's trunc_normal_
def trunc_normal_(tensor, mean=0.0, std=1.0):
    paddle.nn.initializer.TruncatedNormal(mean=mean, std=std)(tensor)


# Replace PyTorch's calculate_gain
def calculate_gain(nonlinearity, param=None):
    linear_gains = {
        'linear': 1.0,
        'conv2d': 1.0,
        'sigmoid': 1.0,
        'tanh': 5.0 / 3,
        'relu': math.sqrt(2.0),
        'leaky_relu': math.sqrt(2.0 / (1 + 0.01 ** 2)) if param is None else math.sqrt(2.0 / (1 + param ** 2)),
    }
    return linear_gains.get(nonlinearity, 1.0)


# Replace PyTorch's DropPath
class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)
        output = x.divide(keep_prob) * random_tensor
        return output


class Convlutioanl(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm2D(out_channel)
        self.relu = nn.ReLU()

    def forward(self, input):
        out = F.pad(input, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Convlutioanl_out(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl_out, self).__init__()
        self.conv = nn.Conv2D(in_channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.tanh = nn.Tanh()

    def forward(self, input):
        out = self.conv(input)
        out = self.tanh(out)
        return out


class SFilterNorm(nn.Layer):
    def __init__(self, filter_type, in_channels=64, kernel_size=3, nonlinearity='linear',
                 running_std=False, running_mean=False):
        super(SFilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) // kernel_size
        if running_std:
            self.std = self.create_parameter(
                shape=[in_channels * kernel_size ** 2],
                default_initializer=nn.initializer.Normal(0, std))
        else:
            self.std = std
        if running_mean:
            self.mean = self.create_parameter(
                shape=[in_channels * kernel_size ** 2],
                default_initializer=nn.initializer.Normal(0, 1.0))

    def forward(self, x):
        b, _, h, w = x.shape
        x = x.reshape([b, self.in_channels, -1, h, w])
        x = x - x.mean(axis=2).reshape([b, self.in_channels, 1, h, w])
        x = x / (x.std(axis=2).reshape([b, self.in_channels, 1, h, w]) + 1e-10)
        x = x.reshape([b, -1, h, w])
        if self.runing_std:
            x = x * self.std.reshape([1, -1, 1, 1])
        else:
            x = x * self.std
        if self.runing_mean:
            x = x + self.mean.reshape([1, -1, 1, 1])
        return x


class CFilterNorm(nn.Layer):
    def __init__(self, filter_type, nonlinearity='linear', in_channels=64, kernel_size=3,
                 running_std=False, running_mean=False):
        assert in_channels >= 1
        super(CFilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = self.create_parameter(
                shape=[in_channels * kernel_size ** 2],
                default_initializer=nn.initializer.Normal(0, std))
        else:
            self.std = std
        if running_mean:
            self.mean = self.create_parameter(
                shape=[in_channels * kernel_size ** 2],
                default_initializer=nn.initializer.Normal(0, 1.0))

    def forward(self, x):
        b = x.shape[0]
        c = self.in_channels
        x = x.reshape([b, c, -1])
        x = x - x.mean(axis=2).reshape([b, c, 1])
        x = x / (x.std(axis=2).reshape([b, c, 1]) + 1e-10)
        x = x.reshape([b, -1])
        if self.runing_std:
            x = x * self.std.reshape([1, -1])
        else:
            x = x * self.std
        if self.runing_mean:
            x = x + self.mean.reshape([1, -1])
        return x


class build_spatial_branch(nn.Layer):
    def __init__(self, channel, reduction=8):
        super(build_spatial_branch, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 3, padding=1),
            nn.Conv2D(channel // reduction, 1, 3, padding=1)
        )

    def forward(self, input):
        return self.body(input)


class build_channel_branch(nn.Layer):
    def __init__(self, channel, nonlinearity='linear', reduction=8, kernel_size=3):
        super(build_channel_branch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Conv2D(channel, 8, 1),
            nn.ReLU(),
            nn.Conv2D(8, channel, 1))

    def forward(self, input):
        out = self.avg_pool(input)
        out = self.fc(out)
        return out


class DDFPack(nn.Layer):
    def __init__(self, in_channels, kernel_size=3, stride=1, dilation=1, head=1,
                 se_ratio=8, nonlinearity='relu', gen_kernel_size=1, kernel_combine='mul'):
        super(DDFPack, self).__init__()
        assert kernel_size > 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.head = head
        self.kernel_combine = kernel_combine

        self.spatial_branch = build_spatial_branch(
            in_channels)

        self.channel_branch = build_channel_branch(
            in_channels, kernel_size, nonlinearity, se_ratio)

    def forward(self, x):
        b, c, h, w = x.shape
        g = self.head
        k = self.kernel_size
        channel_filter = self.channel_branch(x)
        spatial_filter = self.spatial_branch(x)

        XC = x * channel_filter
        XS = x * spatial_filter
        out = XS + XC
        return out


# ====== NEW MODULES FOR OPTIMIZATION ======

# 1. Simplified Linear Attention for SLAB
class SimplifiedLinearAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # Project q, k, v
        q = self.q_proj(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])  # B, h, N, d
        k = self.k_proj(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])  # B, h, N, d
        v = self.v_proj(x).reshape([B, N, self.num_heads, C // self.num_heads]).transpose([0, 2, 1, 3])  # B, h, N, d

        # Apply ELU+1 for numerical stability (key insight from SLAB paper)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Linear attention mechanism (O(N) complexity instead of O(N²))
        # Equivalent to torch.einsum
        kv = paddle.matmul(k.transpose([0, 1, 3, 2]), v)  # B, h, d, d
        qkv = paddle.matmul(q, kv)  # B, h, N, d

        # Normalization factor
        k_sum = paddle.sum(k, axis=2)  # B, h, d
        normalizer = paddle.matmul(q, k_sum.unsqueeze(-1))  # B, h, N, 1
        normalizer = paddle.clip(normalizer, min=1e-6)  # B, h, N, 1
        attn_output = qkv / normalizer

        # Merge heads
        attn_output = attn_output.transpose([0, 2, 1, 3]).reshape([B, N, C])

        # Project to output
        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output


# 2. Progressive Re-parameterized Batch Normalization
class ProgressiveReParamBatchNorm(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # Progressive branch with learnable parameters
        self.alpha = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(0.0))
        self.gamma = self.create_parameter(shape=[dim], default_initializer=nn.initializer.Constant(1.0))
        self.beta = self.create_parameter(shape=[dim], default_initializer=nn.initializer.Constant(0.0))

    def forward(self, x):
        # Standard normalization
        x_norm = self.norm(x)

        # Progressive branch (identity mapping with learnable weights)
        x_id = x * self.alpha

        # Combine with re-parameterization
        out = x_norm * (1.0 - self.alpha) + x_id
        out = out * self.gamma + self.beta

        return out


# Define Mlp class before SLAB as it's used there
class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 3. SLAB Module (to replace Transformer Block)
class SLAB(nn.Layer):
    def __init__(self, dim, input_resolution, num_heads=8, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # First normalization layer (Progressive Re-parameterized Batch Norm)
        self.norm1 = ProgressiveReParamBatchNorm(dim)

        # Simplified Linear Attention
        self.attn = SimplifiedLinearAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Second normalization layer
        self.norm2 = ProgressiveReParamBatchNorm(dim)

        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        B, C, H, W = x.shape

        # Reshape for attention
        x_reshape = x.flatten(2).transpose([0, 2, 1])  # B, H*W, C

        # Apply attention with residual connection
        shortcut = x_reshape
        x_norm = self.norm1(x_reshape)
        x_attn = self.attn(x_norm)
        x_reshape = shortcut + self.drop_path(x_attn)

        # Apply MLP with residual connection
        shortcut = x_reshape
        x_norm = self.norm2(x_reshape)
        x_mlp = self.mlp(x_norm)
        x_reshape = shortcut + self.drop_path(x_mlp)

        # Reshape back
        x = x_reshape.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x


# 4. SLAB Layer (to replace BasicLayer)
class SLABLayer(nn.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Create SLAB blocks - no shift operation needed in SLAB
        self.blocks = nn.LayerList([
            SLAB(dim=dim, input_resolution=input_resolution,
                 num_heads=num_heads, mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


# 5. Enhanced IR Enhancement Module
class IREnhancement(nn.Layer):
    def __init__(self, channel):
        super(IREnhancement, self).__init__()

        # Thermal target enhancement using dilated convolutions for larger receptive field
        self.thermal_enhance = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, channel, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

        # Attention for thermal regions
        self.thermal_attn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, channel // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(channel // 4, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enhanced = self.thermal_enhance(x)
        attn = self.thermal_attn(enhanced)
        return enhanced * attn + x  # Residual connection


# 6. Enhanced VI Enhancement Module with improved texture preservation
class VIEnhancement(nn.Layer):
    def __init__(self, channel):
        super(VIEnhancement, self).__init__()

        # Enhanced texture extraction with multi-scale processing
        self.texture_enhance = nn.Sequential(
            # First branch: standard convolution for global features
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),

            # Second branch: depthwise convolution for texture features
            nn.Conv2D(channel, channel, kernel_size=3, padding=1, groups=channel),
            nn.BatchNorm2D(channel),
            nn.ReLU(),

            # Pointwise convolution to combine features
            nn.Conv2D(channel, channel, kernel_size=1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

        # Dedicated edge enhancement with Sobel-like filters
        self.edge_enhance = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

        # Fine detail preservation branch with smaller kernel
        self.detail_enhance = nn.Sequential(
            nn.Conv2D(channel, channel // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(channel // 2, channel // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2D(channel // 2, channel, kernel_size=1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

    def forward(self, x):
        texture = self.texture_enhance(x)
        edge = self.edge_enhance(x)
        detail = self.detail_enhance(x)

        # Combine all enhancement paths with the input features
        return texture + edge + detail + x  # Multi-residual connection


# 7. Enhanced Cross-Modal Attention Module with greater VI feature influence
class EnhancedCrossModalAttention(nn.Layer):
    def __init__(self, channel):
        super(EnhancedCrossModalAttention, self).__init__()

        # Enhanced channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channel, channel // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2D(channel // 2, channel, kernel_size=1),
            nn.Sigmoid()
        )

        # Enhanced spatial attention with multi-scale awareness
        self.spatial_attn = nn.Sequential(
            # Multi-scale spatial attention
            nn.Conv2D(channel, channel, kernel_size=7, padding=3, groups=channel),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Special processing for VI features to enhance texture details
        self.texture_enhance = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, guide):
        # Extract channel attention from guide branch
        channel_map = self.channel_attn(guide)

        # Process texture details when guide contains VI features
        texture_map = self.texture_enhance(guide)

        # Extract spatial attention from current branch with enhanced focus
        spatial_map = self.spatial_attn(x)

        # Apply enhanced cross-modal attention with texture preservation
        enhanced = x * channel_map * spatial_map + x * texture_map * 0.5

        return enhanced + x  # Residual connection


# 8. Enhanced Density-Adaptive Multi-modal Fusion with improved VI texture preservation
class EnhancedDAMF(nn.Layer):
    def __init__(self, channel):
        super(EnhancedDAMF, self).__init__()

        # Enhanced density estimation
        self.density_conv = nn.Sequential(
            nn.Conv2D(channel, channel // 2, kernel_size=1),
            nn.BatchNorm2D(channel // 2),
            nn.ReLU(),
            nn.Conv2D(channel // 2, 1, kernel_size=1)
        )

        # Enhanced IR feature refinement
        self.refine_ir = nn.Sequential(
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

        # Enhanced VI feature refinement with multi-scale processing for better texture
        self.refine_vi = nn.Sequential(
            # First scale: standard processing
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),

            # Second scale: deeper processing for texture details
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),

            # Extra path for finer details
            nn.Conv2D(channel, channel, kernel_size=1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

        # Multi-scale fusion network
        self.fusion_conv = nn.Sequential(
            nn.Conv2D(channel * 2, channel, kernel_size=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU(),
            nn.Conv2D(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2D(channel),
            nn.ReLU()
        )

    def forward(self, ir_feat, vi_feat):
        # Estimate density-based importance
        ir_density = self.density_conv(ir_feat)
        vi_density = self.density_conv(vi_feat)

        # Create fusion weights with boosted VI influence
        fusion_weights = paddle.concat([ir_density, vi_density * 1.2], axis=1)  # Boost VI weight
        fusion_weights = F.softmax(fusion_weights, axis=1)

        # Split fusion weights for IR and VI
        ir_weight, vi_weight = paddle.split(fusion_weights, 2, axis=1)

        # Refine features with enhanced processing
        ir_refined = self.refine_ir(ir_feat)
        vi_refined = self.refine_vi(vi_feat)

        # Apply weights and combine with higher VI influence
        weighted_ir = ir_refined * ir_weight
        weighted_vi = vi_refined * vi_weight

        # Concatenate and fuse
        fused = paddle.concat([weighted_ir, weighted_vi], axis=1)
        fused = self.fusion_conv(fused)



        return fused


# Original PatchEmbed module (converted to Paddle)
class PatchEmbed(nn.Layer):
    def __init__(self, img_size=120, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


# Modified MODEL class with all optimizations and added parameters
class MODEL(nn.Layer):
    def __init__(self, in_channel=1, out_channel=64, output_channel=1, stride=1, cardinality=1, base_width=64,
                 dilation=1, first_dilation=1, aa_layer=None, se_ratio=8, gen_kernel_size=1,
                 img_size=120, patch_size=4, embed_dim=96, num_heads=8, window_size=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., patch_norm=True, depth=2,
                 downsample=None, drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(MODEL, self).__init__()

        self.convInput = Convlutioanl(in_channel, out_channel)
        self.convolutional_out = Convlutioanl_out(out_channel, output_channel)

        width = int(math.floor(out_channel * (base_width / 64)) * cardinality)
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # DDF module for feature extraction
        self.DDF = DDFPack(width, kernel_size=3, stride=1 if use_aa else stride,
                           dilation=first_dilation, se_ratio=se_ratio,
                           gen_kernel_size=gen_kernel_size, kernel_combine='mul')

        # Enhanced specialized modules for IR and VI branches
        self.ir_enhance = IREnhancement(out_channel)
        self.vi_enhance = VIEnhancement(out_channel)  # Enhanced VI feature extraction

        # Enhanced cross-modal attention modules
        self.cross_attn_ir2vi = EnhancedCrossModalAttention(out_channel)
        self.cross_attn_vi2ir = EnhancedCrossModalAttention(out_channel)

        # Enhanced feature fusion with improved DAMF
        self.fusion = EnhancedDAMF(out_channel)

        # SLAB layer settings
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Replace BasicLayer with SLABLayer (more efficient)
        self.slabLayer = SLABLayer(dim=out_channel,
                                   input_resolution=(patches_resolution[0], patches_resolution[1]),
                                   depth=depth,
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path,
                                   norm_layer=norm_layer,
                                   downsample=downsample,
                                   use_checkpoint=use_checkpoint)

        # 删除最终输出处理模块的引用
        # self.final_process = FinalOutputModule(out_channel)

    def forward(self, ir, vi):
        # IR branch with specialized thermal enhancement
        convInput_A1 = self.convInput(ir)
        layer_A1 = self.DDF(convInput_A1)
        layer_A1 = self.ir_enhance(layer_A1)  # Thermal target enhancement

        # VI branch with enhanced texture extraction
        convInput_B1 = self.convInput(vi)
        layer_B1 = self.DDF(convInput_B1)
        layer_B1 = self.vi_enhance(layer_B1)  # Enhanced texture and detail preservation

        # Enhanced cross-modal attention for mutual feature guidance
        layer_A1_attn = self.cross_attn_ir2vi(layer_A1, layer_B1)
        layer_B1_attn = self.cross_attn_vi2ir(layer_B1, layer_A1)

        # Apply SLAB feature extraction
        encode_size_A1 = (layer_A1_attn.shape[2], layer_A1_attn.shape[3])
        encode_size_B1 = (layer_B1_attn.shape[2], layer_B1_attn.shape[3])
        Transformer_A1 = self.slabLayer(layer_A1_attn, encode_size_A1)
        Transformer_B1 = self.slabLayer(layer_B1_attn, encode_size_B1)

        # Enhanced feature fusion with improved VI texture preservation
        fused = self.fusion(Transformer_A1, Transformer_B1)

        # 删除最终处理步骤，直接将融合结果传递给输出层
        # refined = self.final_process(fused)
        # out = self.convolutional_out(refined)

        # 直接使用融合结果
        out = self.convolutional_out(fused)

        return out