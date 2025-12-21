from re import A
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import einops
import torch.utils.checkpoint
from functools import partial
import open_clip
import numpy as np
from PIL import Image

import torch.nn.functional as F
import timm
from timm.models.layers import trunc_normal_, Mlp
from .sigmoid.module import LayerNorm, RMSNorm, AdaRMSNorm, TDRMSNorm, QKNorm, TimeDependentParameter
from .common_layers import Linear, EvenDownInterpolate, ChannelFirst, ChannelLast, Embedding
from .axial_rope import AxialRoPE, make_axial_pos
from .trans_autoencoder import TransEncoder, Adaptor

def check_zip(*args):
    args = [list(arg) for arg in args]
    length = len(args[0])
    for arg in args:
        assert len(arg) == length
    return zip(*args)
    
class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim_in, dim_out, ratio = 2):
        super().__init__()
        self.ratio = ratio
        self.kernel = Linear(dim_in, dim_out * self.ratio * self.ratio)
    
    def forward(self, x):
        x = self.kernel(x)
        B, H, W, C = x.shape
        x = x.reshape(B, H, W, self.ratio, self.ratio, C // self.ratio // self.ratio)
        x = x.transpose(2, 3)
        x = x.reshape(B, H * self.ratio, W * self.ratio, C // self.ratio // self.ratio)
        return x
    
class PositionEmbeddings(nn.Module):
    def __init__(self, max_height, max_width, dim):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.position_embeddings = Embedding(self.max_height * self.max_width, dim)

    def forward(self, x):
        B, H, W, C = x.shape
        height_idxes = torch.arange(H, device = x.device)[:, None].repeat(1, W)
        width_idxes = torch.arange(W, device = x.device)[None, :].repeat(H, 1)
        idxes = height_idxes * self.max_width + width_idxes
        x = x + self.position_embeddings(idxes[None])
        return x

class TextPositionEmbeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim)
    
    def forward(self, x):
        batch_size, num_embeddings, embedding_dim = x.shape
        # positions = torch.arange(height * width, device=x.device).reshape(1, height, width)
        positions = torch.arange(num_embeddings, device=x.device).unsqueeze(0).expand(batch_size, num_embeddings)
        x = x + self.embedding(positions)
        return x

    
class MLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.norm_type == 'LN':
            self.norm_type = 'LN'
            self.norm = LayerNorm(config.dim)
        elif config.norm_type == 'RMSN':
            self.norm_type = 'RMSN'
            self.norm = RMSNorm(config.dim)
        elif config.norm_type == 'TDRMSN':
            self.norm_type = 'TDRMSN'
            self.norm = TDRMSNorm(config.dim)
        elif config.norm_type == 'ADARMSN':
            self.norm_type = 'ADARMSN'
            self.norm = AdaRMSNorm(config.dim, config.dim)
        self.act = nn.GELU()
        self.w0 = Linear(config.dim, config.hidden_dim)
        self.w1 = Linear(config.dim, config.hidden_dim)
        self.w2 = Linear(config.hidden_dim, config.dim)

    def forward(self, x):
        if self.norm_type == 'LN' or self.norm_type == 'RMSN' or self.norm_type == 'TDRMSN':
            x = self.norm(x)
        elif self.norm_type == 'ADARMSN':
            condition = x[:,0]
            x = self.norm(x, condition)
        x = self.act(self.w0(x)) * self.w1(x)
        x = self.w2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.dim // config.num_attention_heads

        if hasattr(config, "self_att_prompt") and config.self_att_prompt:
            self.condition_key_value = Linear(config.clip_dim, 2 * config.dim, bias = False)

        if config.norm_type == 'LN':
            self.norm_type = 'LN'
            self.norm = LayerNorm(config.dim)
        elif config.norm_type == 'RMSN':
            self.norm_type = 'RMSN'
            self.norm = RMSNorm(config.dim)
        elif config.norm_type == 'TDRMSN':
            self.norm_type = 'TDRMSN'
            self.norm = TDRMSNorm(config.dim)
        elif config.norm_type == 'ADARMSN':
            self.norm_type = 'ADARMSN'
            self.norm = AdaRMSNorm(config.dim, config.dim)

        self.pe_type = config.pe_type
        if config.pe_type == 'Axial_RoPE':
            self.pos_emb = AxialRoPE(self.head_dim, self.num_heads)
            self.qk_norm = QKNorm(self.num_heads)

        self.query_key_value = Linear(config.dim, 3 * config.dim, bias = False)
        self.dense = Linear(config.dim, config.dim)

    def forward(self, x, condition_embeds, condition_masks, pos=None):
        B, N, C = x.shape
        
        if self.norm_type == 'LN' or self.norm_type == 'RMSN' or self.norm_type == 'TDRMSN':
            qkv = self.query_key_value(self.norm(x))
        elif self.norm_type == 'ADARMSN':
            condition = x[:,0]
            qkv = self.query_key_value(self.norm(x, condition))
        q, k, v = qkv.reshape(B, N, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).float().chunk(3, dim = 1)

        if self.pe_type == 'Axial_RoPE':
            q = self.pos_emb(self.qk_norm(q), pos)
            k = self.pos_emb(self.qk_norm(k), pos)

        if condition_embeds is not None:
            _, L, D = condition_embeds.shape
            kcvc = self.condition_key_value(condition_embeds)
            kc, vc = kcvc.reshape(B, L, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).float().chunk(2, dim = 1)
            k = torch.cat([k, kc], dim = 2)
            v = torch.cat([v, vc], dim = 2)
            mask = torch.cat([torch.ones(B, N, dtype = torch.bool, device = condition_masks.device), condition_masks], dim = -1)
            mask = mask[:, None, None, :]
        else:
            mask = None

        x = F.scaled_dot_product_attention(q, k, v, attn_mask = mask)
        x = self.dense(x.permute(0, 2, 1, 3).reshape(B, N, C))

        return x

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.dim % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.dim // config.num_attention_heads

        if config.norm_type == 'LN':
            self.norm_type = 'LN'
            self.norm_q = LayerNorm(config.dim)
            self.norm_kv = LayerNorm(config.dim)
        elif config.norm_type == 'RMSN':
            self.norm_type = 'RMSN'
            self.norm_q = RMSNorm(config.dim)
            self.norm_kv = RMSNorm(config.dim)
        elif config.norm_type == 'TDRMSN':
            self.norm_type = 'TDRMSN'
            self.norm_q = TDRMSNorm(config.dim)
            self.norm_kv = TDRMSNorm(config.dim)
        elif config.norm_type == 'ADARMSN':
            self.norm_type = 'ADARMSN'
            self.norm_q = AdaRMSNorm(config.dim, config.dim)
            self.norm_kv = AdaRMSNorm(config.dim, config.dim)

        self.pe_type = config.pe_type
        if config.pe_type == 'Axial_RoPE':
            self.pos_emb = AxialRoPE(self.head_dim, self.num_heads)
            self.qk_norm = QKNorm(self.num_heads)

        # Query 来自 images (x)
        self.query = Linear(config.dim, config.dim, bias = False)
        # Key 和 Value 来自 image_latent（已经处理成相同维度）
        self.key_value = Linear(config.dim, 2 * config.dim, bias = False)
        
        self.dense = Linear(config.dim, config.dim)

    def forward(self, x, image_latent, pos=None):
        """
        Args:
            x: (B, N, C) - images 的特征，作为 query
            image_latent: (B, N, C) - input image 的 latent（已经处理成序列形式），作为 key 和 value
            pos: 位置编码（用于 query 和 key，因为它们空间对齐）
        """
        B, N, C = x.shape
        B_latent, N_latent, C_latent = image_latent.shape
        assert B == B_latent and N == N_latent and C == C_latent, \
            f"Shape mismatch: x={x.shape}, image_latent={image_latent.shape}"
        
        # Query 来自 x (images)
        if self.norm_type == 'LN' or self.norm_type == 'RMSN' or self.norm_type == 'TDRMSN':
            q = self.query(self.norm_q(x))
        elif self.norm_type == 'ADARMSN':
            condition = x[:,0]
            q = self.query(self.norm_q(x, condition))
        
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).float()

        # Key 和 Value 来自 image_latent
        if self.norm_type == 'LN' or self.norm_type == 'RMSN' or self.norm_type == 'TDRMSN':
            kv = self.key_value(self.norm_kv(image_latent))
        elif self.norm_type == 'ADARMSN':
            condition = image_latent[:,0]
            kv = self.key_value(self.norm_kv(image_latent, condition))
        
        k, v = kv.reshape(B, N, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).float().chunk(2, dim = 1)

        # 位置编码（query 和 key 都需要，因为它们空间对齐）
        if self.pe_type == 'Axial_RoPE':
            q = self.pos_emb(self.qk_norm(q), pos)
            k = self.pos_emb(self.qk_norm(k), pos)  # key 也需要位置编码

        # Cross-attention: q 来自 x, k/v 来自 image_latent
        x = F.scaled_dot_product_attention(q, k, v, attn_mask = None)
        x = self.dense(x.permute(0, 2, 1, 3).reshape(B, N, C))

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 始终创建 SelfAttention（保持架构一致）
        self.block1 = SelfAttention(config)
        
        # 可选地创建 CrossAttention（如果 use_cross_attention=True）
        if hasattr(config, 'use_cross_attention') and config.use_cross_attention:
            self.block1_cross = CrossAttention(config)
            self.use_cross_attention = True
            self.cross_atten_gate = nn.Parameter(torch.zeros(1)) # 初始为0.5
        else:
            self.block1_cross = None
            self.use_cross_attention = False
        
        self.block2 = MLPBlock(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.gradient_checking = config.gradient_checking

    def forward(self, x, condition_embeds=None, condition_masks=None, pos=None, image_latent=None):
        if self.gradient_checking:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, condition_embeds, condition_masks, pos, image_latent
            )
        else:
            return self._forward(x, condition_embeds, condition_masks, pos, image_latent)
    
    def _forward(self, x, condition_embeds=None, condition_masks=None, pos=None, image_latent=None):
        # 1. Self-Attention（始终执行）
        x = x + self.dropout(self.block1(x, condition_embeds, condition_masks, pos))
        
        # 2. Cross-Attention（可选，如果启用且 image_latent 不为 None）
        if self.use_cross_attention and self.block1_cross is not None and image_latent is not None:
            gate = torch.sigmoid(self.cross_atten_gate)
            x = x + gate * self.dropout(self.block1_cross(x, image_latent, pos))
        
        # 3. MLP
        x = x + self.dropout(self.block2(x))
        return x
    
class ConvNeXtBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block1 = nn.Sequential(
            ChannelFirst(), 
            nn.Conv2d(config.dim, config.dim, kernel_size = config.kernel_size, padding = config.kernel_size // 2, stride = 1, groups = config.dim), 
            ChannelLast()
        )
        self.block2 = MLPBlock(config)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.gradient_checking = config.gradient_checking

    def forward(self, x, condition_embeds, condition_masks, pos):
        if self.gradient_checking:
            return torch.utils.checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
    
    def _forward(self, x):
        x = x + self.dropout(self.block1(x))
        x = x + self.dropout(self.block2(x))
        return x
    

class Stage(nn.Module):
    def __init__(self, channels, config, lowres_dim = None, lowres_height = None):
        super().__init__()
        if config.block_type == "TransformerBlock":
            self.encoder_cls = TransformerBlock
        elif config.block_type == "ConvNeXtBlock":
            self.encoder_cls = ConvNeXtBlock
        else:
            raise Exception()
        
        self.pe_type = config.pe_type
        
                
        self.input_layer = nn.Sequential(
            EvenDownInterpolate(config.image_input_ratio),
            nn.Conv2d(channels, config.dim, kernel_size = config.input_feature_ratio, stride = config.input_feature_ratio),
            ChannelLast(),
            PositionEmbeddings(config.max_height, config.max_width, config.dim)
        )

        # 为 image_latent 添加处理层（与 input_layer 类似）
        # 只在需要 cross attention 时创建，避免浪费参数和显存
        if self.encoder_cls is TransformerBlock and hasattr(config, 'use_cross_attention') and config.use_cross_attention:
            self.image_latent_layer = nn.Sequential(
                EvenDownInterpolate(config.image_input_ratio),
                nn.Conv2d(channels, config.dim, kernel_size = config.input_feature_ratio, stride = config.input_feature_ratio),
                ChannelLast(),
                PositionEmbeddings(config.max_height, config.max_width, config.dim)
            )
        else:
            # 不需要 cross attention 或 ConvNeXtBlock，不创建 image_latent_layer
            self.image_latent_layer = None

        if lowres_dim is not None:
            ratio = config.max_height // lowres_height
            self.upsample = nn.Sequential(
                LayerNorm(lowres_dim),
                PixelShuffleUpsample(lowres_dim, config.dim, ratio = ratio),
                LayerNorm(config.dim),
            )

        self.blocks = nn.ModuleList([self.encoder_cls(config) for _ in range(config.num_blocks // 2 * 2 + 1)])
        self.skip_denses = nn.ModuleList([Linear(config.dim * 2, config.dim) for _ in range(config.num_blocks // 2)])

        self.output_layer = nn.Sequential(
            LayerNorm(config.dim),
            ChannelFirst(),
            nn.Conv2d(config.dim, channels, kernel_size = config.final_kernel_size, padding = config.final_kernel_size // 2),
        )

        self.tensor_true = torch.nn.Parameter(torch.tensor([-1.0])) if self.encoder_cls is TransformerBlock else None
        self.tensor_false = torch.nn.Parameter(torch.tensor([1.0])) if self.encoder_cls is TransformerBlock else None


    def forward(self, images, lowres_skips = None, condition_context = None, condition_embeds = None, condition_masks = None, null_indicator=None, image_latent=None):
        if self.pe_type == 'Axial_RoPE' and self.encoder_cls is TransformerBlock:
            x = self.input_layer(images)
            _, H, W, _ = x.shape
            pos = make_axial_pos(H, W)
            
            # 处理 image_latent（如果提供且 image_latent_layer 存在）
            if image_latent is not None and self.image_latent_layer is not None:
                image_latent_processed = self.image_latent_layer(image_latent)
                # image_latent_processed: (B, H, W, C)
                # reshape 成序列形式，与 x 保持一致
                B_latent, H_latent, W_latent, C_latent = image_latent_processed.shape
                assert H == H_latent and W == W_latent, \
                    f"Spatial dimension mismatch: x={x.shape}, image_latent={image_latent_processed.shape}"
                image_latent_seq = image_latent_processed.reshape(B_latent, H_latent * W_latent, C_latent)
            else:
                image_latent_seq = None
        else:
            x = self.input_layer(images)
            pos = None
            # 处理 image_latent（如果提供且 image_latent_layer 存在）
            if image_latent is not None and self.image_latent_layer is not None:
                image_latent_processed = self.image_latent_layer(image_latent)
                B_latent, H_latent, W_latent, C_latent = image_latent_processed.shape
                image_latent_seq = image_latent_processed.reshape(B_latent, H_latent * W_latent, C_latent)
            else:
                image_latent_seq = None

        if lowres_skips is not None:
            x = x + self.upsample(lowres_skips)

        if self.encoder_cls is TransformerBlock:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)

            if null_indicator is not None:
                indicator_tensor = torch.where(null_indicator, self.tensor_true, self.tensor_false)
                indicator_tensor = indicator_tensor.view(B, 1, 1).expand(-1, -1, C)
                x = torch.cat([indicator_tensor, x], dim = 1)
                # image_latent_seq 也需要添加相同的 indicator（如果需要）
                if image_latent_seq is not None:
                    indicator_latent = indicator_tensor  # 使用相同的 indicator
                    image_latent_seq = torch.cat([indicator_latent, image_latent_seq], dim = 1)

        external_skips = [x]

        num_blocks = len(self.blocks)
        in_blocks = self.blocks[:(num_blocks // 2)]
        mid_block = self.blocks[(num_blocks // 2)]
        out_blocks = self.blocks[(num_blocks // 2 + 1):]

        
        skips = []
        for block in in_blocks:
            # 所有 TransformerBlock 都有相同的接口，直接传递参数即可
            # block 内部会根据 use_cross_attention 和 image_latent 决定是否使用 cross attention
            if isinstance(block, TransformerBlock):
                x = block(x, condition_embeds=condition_embeds, condition_masks=condition_masks, pos=pos, image_latent=image_latent_seq)
            else:
                # ConvNeXtBlock 不需要 image_latent 参数
                x = block(x, condition_embeds, condition_masks, pos)
            external_skips.append(x)
            skips.append(x)
        
        # mid_block
        if isinstance(mid_block, TransformerBlock):
            x = mid_block(x, condition_embeds=condition_embeds, condition_masks=condition_masks, pos=pos, image_latent=image_latent_seq)
        else:
            x = mid_block(x, condition_embeds, condition_masks, pos)
        external_skips.append(x)

        for dense, block in check_zip(self.skip_denses, out_blocks):
            x = dense(torch.cat([x, skips.pop()], dim = -1))
            if isinstance(block, TransformerBlock):
                x = block(x, condition_embeds=condition_embeds, condition_masks=condition_masks, pos=pos, image_latent=image_latent_seq)
            else:
                x = block(x, condition_embeds, condition_masks, pos)
            external_skips.append(x)

        if self.encoder_cls is TransformerBlock:

            if null_indicator is not None:
                x = x[:, 1:, :]
                external_skips = [skip[:, 1:, :] for skip in external_skips]
                if image_latent_seq is not None:
                    image_latent_seq = image_latent_seq[:, 1:, :]  # 移除 indicator

            x = x.reshape(B, H, W, C)
            external_skips = [skip.reshape(B, H, W, C) for skip in external_skips]

        output = self.output_layer(x)

        return output, external_skips
    

class MRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.block_grad_to_lowres = config.block_grad_to_lowres
        ########### 新增适配层 ###########

        self.adapter =  nn.Sequential(
        nn.Linear(config.adapter_in_embed, config.clip_dim),
        )
        self.token_compressor = nn.Linear( 576, config.num_clip_token)  # [B, 768, 576] -> [B, 768, 77]

        # nn.Linear(77*2048, 77*768), 全连接降维 [B, 77*2048]输入  [B, 77*768]输出
        ###############################

        for stage_config in config.stage_configs:
            if hasattr(config, "use_t2i"):
                stage_config.use_t2i = config.use_t2i
            if hasattr(config, "clip_dim"):
                stage_config.clip_dim = config.clip_dim
            if hasattr(config, "num_clip_token"):
                stage_config.num_clip_token = config.num_clip_token
            if hasattr(config, "gradient_checking"):
                stage_config.gradient_checking = config.gradient_checking
            if hasattr(config, "pe_type"):
                stage_config.pe_type = config.pe_type
            else:
                stage_config.pe_type = 'APE'
            if hasattr(config, "norm_type"):
                stage_config.norm_type = config.norm_type
            else:
                stage_config.norm_type = 'LN'

        
        #### diffusion model
        if hasattr(config, "not_training_diff") and config.not_training_diff:
            self.has_diff = False
        else:
            self.has_diff = True

            lowres_dims = [None] + [stage_config.dim * (stage_config.num_blocks // 2 * 2 + 2) for stage_config in config.stage_configs[:-1]]
            lowres_heights = [None] + [stage_config.max_height for stage_config in config.stage_configs[:-1]]
            self.stages = nn.ModuleList([
                Stage(self.channels, stage_config, lowres_dim = lowres_dim, lowres_height=lowres_height) 
                for stage_config, lowres_dim, lowres_height in check_zip(config.stage_configs, lowres_dims, lowres_heights)]
            )

        
        #### Text VE
        if hasattr(config.textVAE, "num_down_sample_block"):
            down_sample_block = config.textVAE.num_down_sample_block
        else:
            down_sample_block = 3
        
        self.context_encoder = TransEncoder(d_model=config.clip_dim, N=config.textVAE.num_blocks, num_token=config.num_clip_token,
                                            head_num=config.textVAE.num_attention_heads, d_ff=config.textVAE.hidden_dim, 
                                            latten_size=config.channels*config.stage_configs[-1].max_height*config.stage_configs[-1].max_width * 2, 
                                            down_sample_block=down_sample_block, dropout=config.textVAE.dropout_prob, last_norm=False)



        #### image encoder to train VE
        self.open_clip, _, self.open_clip_preprocess = open_clip.create_model_and_transforms('ViT-L-16-SigLIP-256', pretrained=None)
        if config.stage_configs[-1].max_width==32: 
            # for 256px generation
            self.open_clip_output = Mlp(in_features=1024, 
                                        hidden_features=config.channels*config.stage_configs[-1].max_height*config.stage_configs[-1].max_width, 
                                        out_features=config.channels*config.stage_configs[-1].max_height*config.stage_configs[-1].max_width, 
                                        norm_layer=nn.LayerNorm,
                                    )
        else: 
            # for 512px generation
            self.open_clip_output = Adaptor(input_dim=1024, 
                                        tar_dim=config.channels*config.stage_configs[-1].max_height*config.stage_configs[-1].max_width
                                        )
        del self.open_clip.text
        del self.open_clip.logit_bias
        

    def _forward(self, images, log_snr, condition_context = None, condition_text_embeds = None, condition_text_masks = None, condition_drop_prob = None, null_indicator=None, image_latent=None):
        if self.has_diff:
            TimeDependentParameter.seed_time(self, log_snr)

            assert condition_context is None
            assert condition_text_embeds is None
            
            if condition_text_embeds is not None:
                condition_embeds = self.text_conditioning(condition_text_embeds)
                condition_masks = condition_text_masks
            else:
                condition_embeds = None
                condition_masks = None
            
            outputs = []
            lowres_skips = None
            for stage in self.stages:
                output, lowres_skips = stage(
                    images, 
                    lowres_skips = lowres_skips, 
                    condition_context = condition_context, 
                    condition_embeds = condition_embeds, 
                    condition_masks = condition_masks, 
                    null_indicator=null_indicator,
                    image_latent=image_latent  # 传递 image_latent
                )
                outputs.append(output)
                lowres_skips = torch.cat(lowres_skips, dim = -1)
                if self.block_grad_to_lowres:
                    lowres_skips = lowres_skips.detach()

            return outputs
            
        else:
            return [images]


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def _text_encoder(self, condition_context, tar_shape, mask):

        output = self.context_encoder(condition_context, mask)
#        print("通过VAE之前的shape:",condition_context.shape)
#        print("通过VAE之后的shape:",output.shape)
        mu, log_var = torch.chunk(output, 2, dim=-1)


        z = self._reparameterize(mu, log_var)

        return [z, mu, log_var]
    
    def _text_decoder(self, condition_enbedding, tar_shape):

        context_token = self.context_decoder(condition_enbedding)

        return context_token
    
    def _img_clip(self, image_input):

        image_latent = self.open_clip.encode_image(image_input)
        image_latent = self.open_clip_output(image_latent)

        return image_latent, self.open_clip.logit_scale

        
    
    def forward(self, x, t = None, log_snr = None, text_encoder=False, text_decoder=False, image_clip=False, shape=None, mask=None, null_indicator=None, image_latent=None):
        if text_encoder:
            # 应用适配层 [B, 576, 2048] → [B, 576, 768]
            adapted_cond = self.adapter(x)
            adapted_cond = adapted_cond.transpose(1, 2) # [B, 576, 768] -> [B, 768, 576]
            adapted_cond = self.token_compressor(adapted_cond) # [B, 768, 77]
            adapted_cond = adapted_cond.transpose(1, 2) # [B, 768, 77] -> [B, 77, 768]
            if mask is not None and mask.shape[1] == 576:
                mask = mask[:, :77]
            return self._text_encoder(condition_context = adapted_cond, tar_shape=shape, mask=mask)
        elif text_decoder:
            return self._text_decoder(condition_enbedding = x, tar_shape=shape) # mask is not needed for decoder
        elif image_clip:
            return self._img_clip(image_input = x) 
        else:
            assert log_snr.dtype == torch.float32
            return self._forward(images = x, log_snr = log_snr, null_indicator=null_indicator, image_latent=image_latent)
