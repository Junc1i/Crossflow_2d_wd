"""
This file contains some tools
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import io
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from absl import logging
from PIL import Image, ImageDraw, ImageFont
import textwrap

def save_image_with_caption(image_tensor, caption, filename, font_size=20, font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
    """
    Save an image with a caption
    """
    image_tensor = image_tensor.clone().detach()
    image_tensor = torch.clamp(image_tensor, min=0, max=1)
    image_pil = transforms.ToPILImage()(image_tensor)
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(font_path, font_size)
    wrap_text = textwrap.wrap(caption, width=len(caption)//4 + 1)
    text_sizes = [draw.textsize(line, font=font) for line in wrap_text]
    max_text_width = max(size[0] for size in text_sizes)
    total_text_height = sum(size[1] for size in text_sizes) + 15

    new_height = image_pil.height + total_text_height + 25 
    new_image = Image.new('RGB', (image_pil.width, new_height), 'white')
    new_image.paste(image_pil, (0, 0))
    current_y = image_pil.height + 5
    draw = ImageDraw.Draw(new_image)

    for line, size in zip(wrap_text, text_sizes):
        x = (new_image.width - size[0]) / 2
        draw.text((x, current_y), line, font=font, fill='black')
        current_y += size[1] + 5
    new_image.save(filename)


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'dimr':
        from libs.model.dimr_t2i import MRModel
        return MRModel(kwargs["model_args"])
    elif name == 'dit':
        from libs.model.dit_t2i import DiT_H_2
        return DiT_H_2(kwargs["model_args"])
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, adamw_impl=None, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        impl = (adamw_impl or 'bitsandbytes').lower()
        if impl in ('torch', 'adamw'):
            from torch.optim import AdamW
            return AdamW(params, **kwargs)
        elif impl in ('bitsandbytes', 'adamw8bit'):
            from bitsandbytes.optim import AdamW8bit
            return AdamW8bit(params, **kwargs)
        else:
            raise ValueError(f'Unsupported AdamW implementation: {impl}')
    elif name == 'adafactor':
        from torch.optim import Adafactor
        return Adafactor(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        import shutil
        import time
        import logging
        
        max_retries = 1
        retry_delay = 5  # s
        
        for attempt in range(max_retries):
            temp_path = path + f'.tmp_{int(time.time())}'
            backup_path = path + '.backup'
            
            try:
                # 1. 如果目标路径已存在但不完整，先清理
                if os.path.exists(path):
                    try:
                        # 检查是否是完整的checkpoint（检查step.pth是否存在）
                        if not os.path.exists(os.path.join(path, 'step.pth')):
                            logging.warning(f'Incomplete checkpoint detected at {path}, removing...')
                            shutil.rmtree(path)
                    except Exception as e:
                        logging.warning(f'Error checking checkpoint integrity: {e}')
                
                # 2. 清理当前checkpoint相关的临时目录（只清理自己的）
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                
                # 3. 创建临时目录并保存
                os.makedirs(temp_path, exist_ok=True)
                
                # 4. 保存所有文件到临时目录
                torch.save(self.step, os.path.join(temp_path, 'step.pth'))
                for key, val in self.__dict__.items():
                    if key != 'step' and val is not None:
                        torch.save(val.state_dict(), os.path.join(temp_path, f'{key}.pth'))
                
                # 5. 原子性移动
                if os.path.exists(path):
                    shutil.move(path, backup_path)
                    try:
                        shutil.move(temp_path, path)
                        # 成功后删除备份
                        shutil.rmtree(backup_path)
                    except Exception as e:
                        # 失败则恢复备份
                        if os.path.exists(backup_path):
                            shutil.move(backup_path, path)
                        raise
                else:
                    shutil.move(temp_path, path)
                
                logging.info(f'Successfully saved checkpoint to {path}')
                return  # 成功，退出
                
            except Exception as e:
                logging.warning(f'Save attempt {attempt + 1}/{max_retries} failed: {e}')
                
                # 清理当前操作的临时文件
                for tmp in [temp_path, backup_path]:
                    if os.path.exists(tmp):
                        try:
                            shutil.rmtree(tmp)
                        except:
                            pass
                
                if attempt < max_retries - 1:
                    logging.info(f'Retrying in {retry_delay} seconds...')
                    time.sleep(retry_delay)
                else:
                    logging.error(f'Failed to save checkpoint after {max_retries} attempts')
                    raise

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def trainable_parameters(nnet):
    params_decay = []
    params_nodecay = []
    for name, param in nnet.named_parameters():
        if name.endswith(".nodecay_weight") or name.endswith(".nodecay_bias"):
            params_nodecay.append(param)
        else:
            params_decay.append(param)
    print("params_decay", len(params_decay))
    print("params_nodecay", len(params_nodecay))
    params = [
        {'params': params_decay},
        {'params': params_nodecay, 'weight_decay': 0.0}
    ]
    return params


def initialize_train_state(config, device):

    nnet = get_nnet(**config.nnet)

    if hasattr(config, 'pretrained_path') and config.pretrained_path:
        try:
            # 加载模型参数
            print(f"正在从 {config.pretrained_path} 加载预训练权重...")
            pretrained_dict = torch.load(config.pretrained_path, map_location='cpu')
            model_dict = nnet.state_dict()
            
            # 过滤掉size不匹配的参数,只保留匹配的参数
            matched_dict = {}
            size_mismatch_keys = []
            missing_keys = []
            
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        matched_dict[k] = v
                    else:
                        size_mismatch_keys.append(k)
                        print(f"  ⚠ Size mismatch: {k}")
                        print(f"    预训练: {v.shape}, 当前模型: {model_dict[k].shape}")
            
            # 检查当前模型中缺失的参数
            for k in model_dict.keys():
                if k not in pretrained_dict:
                    missing_keys.append(k)
            
            # 加载匹配的参数
            nnet.load_state_dict(matched_dict, strict=False)
            
            print(f"\n{'='*60}")
            print(f"预训练权重加载报告:")
            print(f"{'='*60}")
            print(f"✓ 成功加载参数数量: {len(matched_dict)}")
            
            # 报告size不匹配的参数
            if size_mismatch_keys:
                print(f"\n⚠ Size不匹配的参数 ({len(size_mismatch_keys)} 个) - 已跳过,将使用随机初始化:")
                for key in size_mismatch_keys[:10]:  # 显示前10个
                    print(f"  • {key}")
                if len(size_mismatch_keys) > 10:
                    print(f"  ... 还有 {len(size_mismatch_keys)-10} 个")
            
            # 处理缺失的参数
            if missing_keys:
                print(f"\n⚠ 缺失参数 ({len(missing_keys)} 个):")
                adapter_keys = [k for k in missing_keys if "adapter" in k]
                other_missing = [k for k in missing_keys if "adapter" not in k]
                
                if adapter_keys:
                    print(f"  - Adapter 相关参数 ({len(adapter_keys)} 个): 将随机初始化")
                    for key in adapter_keys[:5]:  # 只显示前5个
                        print(f"    • {key}")
                    if len(adapter_keys) > 5:
                        print(f"    ... 还有 {len(adapter_keys)-5} 个")
                
                if other_missing:
                    print(f"  - 其他缺失参数 ({len(other_missing)} 个): 将使用默认初始化")
                    for key in other_missing[:5]:  # 只显示前5个
                        print(f"    • {key}")
                    if len(other_missing) > 5:
                        print(f"    ... 还有 {len(other_missing)-5} 个")
            
            print(f"{'='*60}\n")
            
            # 初始化 adapter 层
            if hasattr(nnet, 'adapter'):
                nn.init.xavier_uniform_(nnet.adapter[0].weight)
                if hasattr(nnet.adapter[0], 'bias') and nnet.adapter[0].bias is not None:
                    nn.init.zeros_(nnet.adapter[0].bias)
                print("✓ Adapter layer 初始化完成 (Xavier uniform)")
            
        except FileNotFoundError:
            print(f"\n❌ 错误：找不到预训练权重文件 '{config.pretrained_path}'")
            print("请检查路径是否正确,或者注释掉 config.pretrained_path 从头训练")
            raise
        
        except Exception as e:
            print(f"\n❌ 加载预训练权重时发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("⚠ 未指定预训练权重路径,将从头开始训练(随机初始化)")
        

    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()

    optimizer = get_optimizer(trainable_parameters(nnet), **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, return_clipScore=False, ClipSocre_model=None, config=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    clip_score_list = []

    if return_clipScore:
        assert ClipSocre_model is not None

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = sample_fn(mini_batch_size, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, config=config)
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        #clip_score_list.append(accelerator.gather(clip_score)[:_batch_size])
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1
        
    if return_clipScore:
        return clip_score_list
    else:
        return None


def sample2dir_wCLIP(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, return_clipScore=False, ClipSocre_model=None, config=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    clip_score_list = []

    if return_clipScore:
        assert ClipSocre_model is not None

    for _batch_size in amortize(n_samples, batch_size):
        samples, clip_score = sample_fn(mini_batch_size, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, config=config)
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        clip_score_list.append(accelerator.gather(clip_score)[:_batch_size])
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1
        break
        
    if return_clipScore:
        return clip_score_list
    else:
        return None


def sample2dir_wPrompt(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, config=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    
    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples, samples_caption = sample_fn(mini_batch_size, return_caption=True, config=config)
        samples = unpreprocess_fn(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample, caption in zip(samples,samples_caption):
                try:
                    save_image_with_caption(sample, caption, os.path.join(path, f"{idx}.png"))
                except:
                    save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def sample2dir_with_gt(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None, return_clipScore=False, ClipSocre_model=None, config=None):
    """
    保存生成的图片和对应的ground truth图片，水平拼接后保存
    
    Args:
        accelerator: accelerate.Accelerator 实例
        path: 保存路径
        n_samples: 总样本数
        mini_batch_size: 每个进程的批次大小
        sample_fn: 采样函数，返回 (generated_samples, gt_images)
        unpreprocess_fn: 反预处理函数
        return_clipScore: 是否返回CLIP分数
        ClipSocre_model: CLIP分数模型
        config: 配置对象
    """
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes
    clip_score_list = []

    if return_clipScore:
        assert ClipSocre_model is not None

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir_with_gt'):
        samples, gt_images = sample_fn(mini_batch_size, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, config=config)
        
        # 反预处理生成的图片
        samples = unpreprocess_fn(samples)
        
        # 反预处理ground truth图片（如果还没有unpreprocess）
        # 注意：gt_images 可能已经是 [-1, 1] 范围的tensor，需要unpreprocess
        gt_images = unpreprocess_fn(gt_images)
        
        # 使用 accelerator.gather 收集所有进程的结果
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        gt_images = accelerator.gather(gt_images.contiguous())[:_batch_size]
        
        if accelerator.is_main_process:
            target_size = 256  # 统一调整到 256x256
            
            for sample, gt in zip(samples, gt_images):
                # 确保 sample 和 gt 的形状都是 [C, H, W]
                # 如果高度或宽度不是 256，则 resize 到 256x256
                if sample.shape[1] != target_size or sample.shape[2] != target_size:
                    # sample 需要添加 batch 维度进行 interpolate: [C, H, W] -> [1, C, H, W]
                    sample = sample.unsqueeze(0)
                    sample = F.interpolate(sample, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    sample = sample.squeeze(0)  # 移除 batch 维度: [1, C, H, W] -> [C, H, W]
                
                if gt.shape[1] != target_size or gt.shape[2] != target_size:
                    # gt 需要添加 batch 维度进行 interpolate: [C, H, W] -> [1, C, H, W]
                    gt = gt.unsqueeze(0)
                    gt = F.interpolate(gt, size=(target_size, target_size), mode='bilinear', align_corners=False)
                    gt = gt.squeeze(0)  # 移除 batch 维度: [1, C, H, W] -> [C, H, W]
                
                # 使用 make_grid 来排列图片：将 sample 和 gt 放在一行，添加 padding
                # 此时 sample 和 gt 的形状都是 [C, 256, 256]
                images_pair = torch.stack([sample, gt], dim=0)  # [2, C, 256, 256]
                # nrow=2 表示每行2张图片，padding=2 表示图片间距为2像素，pad_value=1.0 表示填充白色
                concatenated = make_grid(images_pair, nrow=2, padding=2, pad_value=1.0)  # [C, H, W']
                save_image(concatenated, os.path.join(path, f"{idx}.png"))
                idx += 1
        
    if return_clipScore:
        return clip_score_list
    else:
        return None


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


# 全局缓存，避免重复执行tokenizer.encode
_tokenizer_cache = {}
_tokenizer_cache_lock = None

def _get_tokenizer_cache_key(vl_chat_processor, question):
    """生成缓存键"""
    # 使用 question 和 processor 的配置生成唯一键
    cache_key = (
        question,
        vl_chat_processor.sft_format,
        vl_chat_processor.system_prompt,
        id(vl_chat_processor.tokenizer)  # 使用tokenizer对象id确保唯一性
    )
    return cache_key

def _get_or_encode_tokenizer(vl_chat_processor, question, device):
    """获取或编码tokenizer结果（带缓存）"""
    global _tokenizer_cache, _tokenizer_cache_lock
    import threading
    
    # 初始化锁（仅第一次）
    if _tokenizer_cache_lock is None:
        _tokenizer_cache_lock = threading.Lock()
    
    # 生成缓存键
    cache_key = _get_tokenizer_cache_key(vl_chat_processor, question)
    
    # 尝试从缓存获取
    with _tokenizer_cache_lock:
        if cache_key in _tokenizer_cache:
            # 缓存命中，直接返回缓存的input_ids列表（CPU上）
            return _tokenizer_cache[cache_key]
    
    # 缓存未命中，需要编码
    # 创建统一的conversation格式(所有图像使用相同的prompt)
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question}"},
            {"role": "<|Assistant|>", "content": ""},
        ],
        sft_format=vl_chat_processor.sft_format,
        system_prompt=vl_chat_processor.system_prompt,
    )
    
    # tokenize(在CPU上执行，返回Python列表)
    input_ids = vl_chat_processor.tokenizer.encode(sft_format)
    
    # 存入缓存（存储CPU上的列表，避免GPU内存占用）
    with _tokenizer_cache_lock:
        _tokenizer_cache[cache_key] = input_ids
    
    return input_ids

def get_input_image_embeddings_and_masks(
    batch_input_images,  # 接收已预处理的 tensor [batch_size, 3, H, W] 或路径列表（向后兼容）
    vl_chat_processor,
    vl_gpt,
    device,
    question="",
    num_image_tokens=576,
    output_tokens=None,
    accelerator=None,
    cached_input_ids=None  # 新增：可选的预编码input_ids（用于进一步优化）
):
    """
    批量处理输入图像并获取token embeddings和masks
    
    Args:
        batch_input_images: 可能是以下类型之一：
            - torch.Tensor: 已经预处理过的 tensor [batch_size, 3, H, W]（WebDataset 优化模式）
            - 字符串列表: 图像路径列表（文件系统模式，向后兼容）
        vl_chat_processor: Janus的VLChatProcessor实例
        vl_gpt: Janus的MultiModalityCausalLM模型
                注意: 如果模型被accelerator.prepare()包装过,需要先unwrap:
                vl_gpt = accelerator.unwrap_model(vl_gpt) if hasattr(accelerator, 'unwrap_model') else vl_gpt
        device: 设备(torch.device)
        question: 可选的文本问题(默认为空字符串)
        num_image_tokens: 每个图像的token数量(默认576)
        output_tokens: 返回的token数量,如果指定则截取前N个token(默认None,返回全部)
        accelerator: 可选的accelerator实例,用于错误日志中的rank信息
        cached_input_ids: 可选的预编码input_ids列表（CPU上），如果提供则跳过tokenizer.encode
    
    Returns:
        batch_embeddings: torch.Tensor,形状为[batch_size, output_tokens or num_image_tokens, hidden_dim],在指定device上
        batch_attention_masks: torch.Tensor,形状为[batch_size, output_tokens or num_image_tokens],在指定device上
    """
    batch_embeddings_list = []
    batch_attention_masks_list = []
    
    # 检查是否已经是处理过的 tensor
    if isinstance(batch_input_images, torch.Tensor):
        # 已经是处理过的 tensor，直接使用
        # batch_input_images shape: [batch_size, 3, H, W]
        # 如果已经在目标设备上，则不需要移动（避免不必要的传输）
        if batch_input_images.device != device:
            batched_pixel_values = batch_input_images.to(device, non_blocking=True)
        else:
            batched_pixel_values = batch_input_images
        batch_size = batched_pixel_values.shape[0]
    else:
        # 向后兼容：处理路径列表（文件系统模式）
        import concurrent.futures
        
        def load_image(image_input):
            """加载单个图像，支持路径字符串"""
            if isinstance(image_input, str):
                try:
                    pil_img = Image.open(image_input)
                    pil_img.load()
                    return pil_img.convert('RGB')
                except Exception as e:
                    rank_info = f"[Rank {accelerator.process_index}] " if accelerator is not None else ""
                    print(f"{rank_info}警告：加载输入图像失败 {image_input}: {e}")
                    return Image.new('RGB', (384, 384), color='black')
            else:
                rank_info = f"[Rank {accelerator.process_index}] " if accelerator is not None else ""
                print(f"{rank_info}警告：不支持的类型 {type(image_input)}")
                return Image.new('RGB', (384, 384), color='black')
        
        # 加载图像
        if len(batch_input_images) > 0:
            max_workers = min(len(batch_input_images), os.cpu_count() or 1)
            if max_workers > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    all_pil_images = list(executor.map(load_image, batch_input_images))
            else:
                all_pil_images = [load_image(path) for path in batch_input_images]
        else:
            all_pil_images = []
        
        # 批量图像预处理
        images_outputs = vl_chat_processor.image_processor(all_pil_images, return_tensors="pt")
        batched_pixel_values = images_outputs.pixel_values.to(device, non_blocking=True)  # [batch_size, 3, H, W]
        batch_size = len(all_pil_images)
    
    # Step 2: 准备批量输入(文本部分) - 使用缓存优化
    if cached_input_ids is not None:
        # 使用预编码的input_ids（从外部传入，通常是在训练开始前预编码）
        input_ids = cached_input_ids
    else:
        # 使用内部缓存机制（首次调用时编码，后续自动从缓存获取）
        input_ids = _get_or_encode_tokenizer(vl_chat_processor, question, device)
    
    # 直接在GPU上创建tensor，避免CPU到GPU的同步传输
    # 使用torch.empty + fill_ 或直接创建，根据batch_size高效创建
    batched_input_ids = torch.tensor([input_ids] * batch_size, dtype=torch.long, device=device)  # [batch_size, seq_len]
    
    # 创建image masks（直接在GPU上操作）
    image_token_mask = batched_input_ids == vl_chat_processor.image_id
    batched_images_seq_mask = image_token_mask
    
    # 假设每个图像有num_image_tokens个token（直接在GPU上创建）
    batched_images_emb_mask = torch.zeros((batch_size, 1, num_image_tokens), dtype=torch.bool, device=device)
    batched_images_emb_mask[:, :, :num_image_tokens] = True
    
    # 调整pixel_values形状以匹配期望的 [batch_size, n_images, 3, H, W]
    batched_pixel_values = batched_pixel_values.unsqueeze(1)  # [batch_size, 1, 3, H, W]
    
    # Step 4: 批量编码（所有输入已经在GPU上，无需再移动）
    with torch.no_grad():
        inputs_embeds = vl_gpt.prepare_inputs_embeds(
            input_ids=batched_input_ids,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask
        )
    
    # inputs_embeds.shape 可能是 [batch_size, seq_len, hidden_dim] 或 [batch_size, num_image_tokens, hidden_dim]
    # 保持在GPU上,不转换到CPU
    inputs_embeds = inputs_embeds.detach().float()
    
    # 批量提取图像embeddings（优化：避免循环，直接批量操作）
    if inputs_embeds.shape[1] == num_image_tokens:
        # 如果已经是图像部分的embeddings,直接使用
        batch_embeddings = inputs_embeds  # [batch_size, num_image_tokens, hidden_dim]
    else:
        # 从完整序列中批量提取图像token对应的embeddings
        # batched_images_seq_mask: [batch_size, seq_len]
        # inputs_embeds: [batch_size, seq_len, hidden_dim]
        
        # 计算每个样本中图像token的数量（应该都是num_image_tokens，但为了安全起见）
        num_image_tokens_per_sample = batched_images_seq_mask.sum(dim=1)  # [batch_size]
        
        # 检查是否所有样本的图像token数量一致
        if (num_image_tokens_per_sample == num_image_tokens).all():
            # 所有样本的图像token数量一致，可以批量提取
            # 使用mask直接索引（需要处理变长序列）
            batch_embeddings_list = []
            for i in range(batch_size):
                image_mask = batched_images_seq_mask[i]  # [seq_len]
                image_embeddings = inputs_embeds[i][image_mask]  # [num_image_tokens, hidden_dim]
                batch_embeddings_list.append(image_embeddings)
            batch_embeddings = torch.stack(batch_embeddings_list, dim=0)  # [batch_size, num_image_tokens, hidden_dim]
        else:
            # 图像token数量不一致，需要逐个处理并填充/截断
            batch_embeddings_list = []
            for i in range(batch_size):
                image_mask = batched_images_seq_mask[i]  # [seq_len]
                image_embeddings = inputs_embeds[i][image_mask]  # [actual_tokens, hidden_dim]
                
                # 确保形状正确
                if image_embeddings.shape[0] > num_image_tokens:
                    image_embeddings = image_embeddings[:num_image_tokens]
                elif image_embeddings.shape[0] < num_image_tokens:
                    # 填充到num_image_tokens
                    padding = torch.zeros(
                        (num_image_tokens - image_embeddings.shape[0], image_embeddings.shape[1]),
                        device=device,
                        dtype=image_embeddings.dtype
                    )
                    image_embeddings = torch.cat([image_embeddings, padding], dim=0)
                
                batch_embeddings_list.append(image_embeddings)
            batch_embeddings = torch.stack(batch_embeddings_list, dim=0)  # [batch_size, num_image_tokens, hidden_dim]
    
    # 创建attention masks
    batch_attention_masks = torch.ones(
        (batch_size, num_image_tokens), 
        device=device, 
        dtype=torch.long
    )  # [batch_size, num_image_tokens]
    
    # 如果指定了output_tokens,截取前N个token
    if output_tokens is not None and output_tokens < num_image_tokens:
        batch_embeddings = batch_embeddings[:, :output_tokens, :]  # [batch_size, output_tokens, hidden_dim]
        batch_attention_masks = batch_attention_masks[:, :output_tokens]  # [batch_size, output_tokens]
    
    return batch_embeddings, batch_attention_masks