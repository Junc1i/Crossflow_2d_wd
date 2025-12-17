import ml_collections
import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import os
import sys
import wandb
import numpy as np
import time
import random
from PIL import Image

# 抑制非关键警告
import warnings
warnings.filterwarnings("ignore", message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.GradScaler.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Importing from timm.models.layers.*")
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*use_reentrant.*")
warnings.filterwarnings("ignore", message=".*Using slower tdp_torch implementation.*")


# 设置新的 NCCL 环境变量
if 'NCCL_ASYNC_ERROR_HANDLING' in os.environ:
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = os.environ['NCCL_ASYNC_ERROR_HANDLING']

import libs.autoencoder
from libs.t5 import T5Embedder
from libs.clip import FrozenCLIPEmbedder
from diffusion.flow_matching import FlowMatching, ODEFlowMatchingSolver, ODEEulerFlowMatchingSolver
from tools.fid_score import calculate_fid_given_paths
from tools.clip_score import ClipSocre
import webdataset as wds

original_sys_path = sys.path.copy()
crossflow_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
janus_dir = os.path.join(crossflow_parent_dir, "Janus")
if janus_dir not in sys.path:
    sys.path.insert(0, janus_dir) 
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
sys.path = original_sys_path
from transformers import AutoModelForCausalLM

def train(config):
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=1)
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # WandB 配置：支持命令行参数、配置文件或环境变量
        # wandb_mode 优先级：命令行参数 > 配置文件 > 环境变量 > 默认值(online)
        wandb_mode = (
            getattr(config, 'wandb_mode', None) or  # 命令行参数或配置文件
            os.environ.get('WANDB_MODE', None) or   # 环境变量
            'online'  # 默认值
        )
        
        # wandb_project 优先级：命令行参数 > 配置文件 > 环境变量 > 默认命名
        wandb_project = (
            getattr(config, 'wandb_project', None) or  # 命令行参数或配置文件
            os.environ.get('WANDB_PROJECT', None) or   # 环境变量
            f'{config.config_name}_{config.dataset.name}'  # 默认：配置名_数据集名
        )
        
        wandb.init(dir=os.path.abspath(config.workdir), project=wandb_project, config=config.to_dict(),
                   name=config.hparams, job_type='train', mode=wandb_mode)
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
        logging.info(f'Optimizer config: {config.optimizer}')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    # 定义 num_workers（可以从配置中读取，或使用默认值）
    gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
    num_workers = getattr(config, 'num_workers', 8)
    
    # 传递参数到 dataset（包括 fid_stat_path, num_workers, batch_size）
    dataset_kwargs = dict(config.dataset)
    if hasattr(config, 'fid_stat_path') and config.fid_stat_path:
        dataset_kwargs['fid_stat_path'] = config.fid_stat_path
    
    # 传递 num_workers 和 batch_size 给数据集
    dataset_kwargs['num_workers'] = num_workers
    dataset_kwargs['batch_size'] = mini_batch_size  # 每个进程的 batch_size
    dataset = get_dataset(**dataset_kwargs)

    # 提前加载模型，以便在创建 DataLoader 之前设置 processor
    model_path = "deepseek-ai/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    # 预编码tokenizer（优化：避免在训练循环中重复执行tokenizer.encode）
    training_question = ""
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=[
            {"role": "<|User|>", "content": f"<image_placeholder>\n{training_question}"},
            {"role": "<|Assistant|>", "content": ""},
        ],
        sft_format=vl_chat_processor.sft_format,
        system_prompt=vl_chat_processor.system_prompt,
    )
    # 预编码input_ids（CPU上，仅执行一次）
    cached_training_input_ids = vl_chat_processor.tokenizer.encode(sft_format)
    logging.info(f'预编码tokenizer完成, input_ids长度: {len(cached_training_input_ids)}')

    train_dataset = dataset.get_split(split='train', labeled=True)
    
    # WebDataset 不支持索引访问，需要条件判断
    from datasets import WebDatasetDataset
    is_webdataset = isinstance(train_dataset, WebDatasetDataset)
    
    # 如果使用 WebDataset，在创建 DataLoader 之前设置 vl_chat_processor
    if is_webdataset:
        train_dataset.set_vl_chat_processor(vl_chat_processor)
        train_dataset.set_device(device)  # 使用 accelerator.device
    
    # WebDataset 需要使用 wds.WebLoader
    if is_webdataset:
        train_dataset_loader = wds.WebLoader(
            train_dataset,
            batch_size=None,  # batch_size 已经在 pipeline 中通过 wds.batched() 处理
            shuffle=False,  # WebDataset 已经在 pipeline 中 shuffle
            num_workers=num_workers,  # 可以使用多个 worker 加速数据加载
            pin_memory=True,  # 使用 pin_memory 加速 CPU 到 GPU 的传输
            persistent_workers=True if num_workers > 0 else False,  # num_workers > 0 时可以使用 persistent_workers
        )
        # wds.WebLoader 不支持 drop_last，因为批处理已经在 pipeline 中完成
    else:
        # 传统 Dataset 模式
        train_dataset_loader = DataLoader(
            train_dataset, 
            batch_size=mini_batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True,
            prefetch_factor=4,
        )

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    is_test_webdataset = isinstance(test_dataset, WebDatasetDataset)
    
    # 如果使用 WebDataset，在创建 DataLoader 之前设置 vl_chat_processor
    if is_test_webdataset:
        test_dataset.set_vl_chat_processor(vl_chat_processor)
        test_dataset.set_device(device)  # 使用 accelerator.device
    
    if is_test_webdataset:
        test_dataset_loader = wds.WebLoader(
            test_dataset,
            batch_size=None,  # batch_size 已经在 pipeline 中通过 wds.batched() 处理
            shuffle=False,
            num_workers=4,  # 可以使用多个 worker 加速数据加载
            pin_memory=True,  # 使用 pin_memory 加速 CPU 到 GPU 的传输（数据在 CPU 上）
            persistent_workers=True if num_workers > 0 else False,  # num_workers > 0 时可以使用 persistent_workers
        )
    else:
        test_dataset_loader = DataLoader(
            test_dataset, 
            batch_size=config.sample.mini_batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=num_workers, 
            pin_memory=True, 
            persistent_workers=True,
        )


    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    # 使用 resume_ckpt_root 进行 resume，保存时仍然使用 ckpt_root
    # 支持两种格式：
    # 1. checkpoint 目录路径（如 /path/to/ckpts/60000.ckpt，其中 60000.ckpt 是目录）
    # 2. ckpt_root 目录路径（如 /path/to/ckpts，会在其中查找最新的 checkpoint）
    resume_path = config.resume_ckpt_root
    if resume_path and resume_path.endswith('.ckpt') and os.path.isdir(resume_path):
        # 如果指定的是 checkpoint 目录路径（以 .ckpt 结尾且是目录），直接使用 load
        logging.info(f'从 checkpoint 目录直接加载: {resume_path}')
        train_state.load(resume_path)
    else:
        # 否则使用 resume，会在目录中查找最新的 checkpoint
        train_state.resume(resume_path)

#    for param in nnet.parameters():
#        print("para.device:",param.device) 

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )
    vl_gpt = vl_gpt.half().eval().to(device)

    # 修改为：
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        use_safetensors=True,
        device_map={
            "vision_model": 0,      # GPU 0
            "aligner": 0,           # GPU 0
            "language_model": "cpu",  # 放到CPU，不占用GPU显存
            "gen_vision_model": "cpu",
            "gen_aligner": "cpu",
            "gen_head": "cpu",
            "gen_embed": "cpu"
        },
        torch_dtype=torch.float16
    )
    vl_gpt.eval()  # 不需要再 .to(device)

    if config.nnet.model_args.clip_dim == 4096:
        llm = "Janus-Pro-7B"
        # t5 = T5Embedder(device=device)
        print("config.nnet.model_args.clip_dim:",config.nnet.model_args.clip_dim)
        print("Using Janus-Pro-7B")
    elif config.nnet.model_args.clip_dim == 768:
        llm = "Janus-Pro-1B"
        print("Using Janus-Pro-1B")
    else:
        raise NotImplementedError

    ss_empty_context = None

    ClipSocre_model = ClipSocre(device=device)

    #@ torch.cuda.amp.autocast()
    #def encode(_batch):
    #    return autoencoder.encode(_batch)

    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            remaining_steps = config.train.n_steps - train_state.step
            for data in tqdm(
                train_dataset_loader,
                disable=not accelerator.is_main_process,
                desc=f'step {train_state.step}/{config.train.n_steps}',
                unit=' its',
                ncols=120,
                dynamic_ncols=True,
                total=remaining_steps,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in tqdm(
                test_dataset_loader,
                disable=not accelerator.is_main_process,
                desc='step',
                unit=' its'
            ):
                yield data

    context_generator = get_context_generator()

    # 获取分布式训练参数
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    
    # 创建 FlowMatching，传递 world_size 和 rank 以支持多GPU特征收集
    _flow_mathcing_model = FlowMatching(world_size=world_size, rank=rank)
    
    # 调试信息
    if accelerator.is_main_process:
        logging.info(f"FlowMatching initialized with world_size={world_size}, rank={rank}")
        logging.info(f"ClipLoss will use multi-GPU feature gathering: {world_size > 1}")

    def train_step(_batch, _ss_empty_context):
        _metrics = dict()
        optimizer.zero_grad()

        assert len(_batch)==3
        assert not config.dataset.cfg
        # WebDataset 模式：
        #   _batch[0]: input 图像 tensor [batch_size, 3, H, W]（CPU上，已通过vl_chat_processor预处理）
        #   _batch[1]: output 图像 tensor [batch_size, C, H, W]（CPU上，已归一化到[-1, 1]）
        #   _batch[2]: input 图像 tensor [batch_size, C, H, W]（CPU上，已归一化到[-1, 1]，用于autoencoder编码）
        # 文件系统模式：
        #   _batch[0]: 图像路径列表（字符串列表）
        #   _batch[1]: output 图像 tensor [batch_size, C, H, W]（CPU上，已归一化到[-1, 1]）
        #   _batch[2]: input 图像 tensor [batch_size, C, H, W]（CPU上，已归一化到[-1, 1]，用于autoencoder编码）
        _batch_input_img = _batch[0]  
        _batch_output_img = _batch[1] # output image tensor
        _batch_input_img_tensor = _batch[2] # input image tensor for autoencoder
        
        # 确保输出图像在GPU上（DataLoader可能返回CPU tensor）
        if isinstance(_batch_output_img, torch.Tensor) and _batch_output_img.device != device:
            _batch_output_img = _batch_output_img.to(device, non_blocking=True)
        
        # 确保输入图像tensor在GPU上
        if isinstance(_batch_input_img_tensor, torch.Tensor) and _batch_input_img_tensor.device != device:
            _batch_input_img_tensor = _batch_input_img_tensor.to(device, non_blocking=True)
        
        # 确保 batch_output_img 有正确的 batch 维度
        # 当 mini_batch_size=1 时，wds.batched 可能返回 [C, H, W] 而不是 [1, C, H, W]
        if isinstance(_batch_output_img, torch.Tensor) and _batch_output_img.dim() == 3:
            _batch_output_img = _batch_output_img.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        # 确保 batch_input_img_tensor 有正确的 batch 维度
        if isinstance(_batch_input_img_tensor, torch.Tensor) and _batch_input_img_tensor.dim() == 3:
            _batch_input_img_tensor = _batch_input_img_tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
        
        moments_256 = autoencoder(_batch_output_img, fn='encode_moments')
        # moments_256 应该是 [batch_size, 8, H, W] 形状，需要保持batch维度
        moments_256 = moments_256.detach()  # 保持为 Tensor，不要转换为 numpy

        _z = autoencoder.sample(moments_256) # 传入moments，返回 [batch_size, 4, H, W]
        
        # 处理 input image 的 latent（用于 cross attention）
        input_moments_256 = autoencoder(_batch_input_img_tensor, fn='encode_moments')
        input_moments_256 = input_moments_256.detach()
        _input_image_latent = autoencoder.sample(input_moments_256)  # [batch_size, 4, H, W]
        
        # 调用函数获取embeddings和masks（直接返回tensor格式，截取前77个token）
        # 注意：_batch_input_img 可能是 PIL Image 列表（WebDataset）或路径列表（文件系统）
        # utils函数会自动检测类型并处理
        # 优化：使用预编码的input_ids，避免每次训练步骤都执行tokenizer.encode
        _batch_con, _batch_mask = utils.get_input_image_embeddings_and_masks(
            batch_input_images=_batch_input_img,  # PIL Image 列表或路径列表
            vl_chat_processor=vl_chat_processor,
            vl_gpt=vl_gpt,
            device=device,
            question="",
            num_image_tokens=576,
            output_tokens=576,  # 返回前77个token: [batch_size, 77, 2048], [batch_size, 77]
            accelerator=accelerator,
            cached_input_ids=cached_training_input_ids  # 传入预编码的input_ids，跳过tokenizer.encode
        )
        with accelerator.accumulate(nnet):    
            loss, loss_dict = _flow_mathcing_model(_z, nnet, loss_coeffs=config.loss_coeffs, cond=_batch_con, con_mask=_batch_mask, batch_img_clip=_batch_output_img, \
            nnet_style=config.nnet.name, text_token=None, model_config=config.nnet.model_args, all_config=config, training_step=train_state.step, image_latent=_input_image_latent)


            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            for key in loss_dict.keys():
                _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
            accelerator.backward(loss.mean())
            optimizer.step()
            lr_scheduler.step()
            train_state.ema_update(config.get('ema_rate', 0.9999))
            train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, context=None, caption=None, testbatch_img_blurred=None, two_stage_generation=-1, token_mask=None, return_clipScore=False, ClipSocre_model=None, image_latent=None):
        with torch.no_grad():
            _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)
                
            _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = _z_gaussian.shape, mask=token_mask)
            _z_init = _z_x0.reshape(_z_gaussian.shape)
            
            assert config.sample.scale > 1
            _cfg = config.sample.scale

            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg)
            _z, _ = ode_solver.sample(x_T=_z_init, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator, image_latent=image_latent)

            image_unprocessed = decode(_z)

            if return_clipScore:
                clip_score = ClipSocre_model.calculate_clip_score(caption, image_unprocessed)
                return image_unprocessed, clip_score
            else:
                return image_unprocessed

    def eval_step(n_samples, sample_steps):
        ###########################################################
        # 确保n_samples不超过测试数据集的大小
        # WebDatasetDataset 不支持 len()，需要特殊处理
        if is_test_webdataset:
            # WebDataset 模式：使用 epoch_size 属性（如果存在）
            if hasattr(test_dataset, 'epoch_size'):
                test_dataset_size = test_dataset.epoch_size
                if n_samples > test_dataset_size:
                    logging.warning(f"n_samples ({n_samples}) 超过测试数据集大小 ({test_dataset_size})，将使用测试数据集大小")
                    n_samples = test_dataset_size
            else:
                # 如果没有 epoch_size 属性，跳过大小检查（WebDataset 是迭代器，无法准确知道大小）
                logging.info(f"WebDataset 模式：跳过数据集大小检查，使用 n_samples={n_samples}")
        else:
            # 传统 Dataset 模式：使用 len()
            test_dataset_size = len(test_dataset)
            if n_samples > test_dataset_size:
                logging.warning(f"n_samples ({n_samples}) 超过测试数据集大小 ({test_dataset_size})，将使用测试数据集大小")
                n_samples = test_dataset_size
        ###########################################################
            
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')
        def sample_fn(_n_samples, return_caption=False, return_clipScore=False, ClipSocre_model=None, config=None):
            _input_img, _output_img, _input_img_tensor = next(context_generator) 
            # _input_img 可能是 PIL Image 列表（WebDataset）或路径列表（文件系统）
            # _output_img 是 output 图像 tensor，形状为 [batch_size, 3, H, W] 或 [batch_size, C, H, W]
            # _input_img_tensor 是 input 图像 tensor，用于 autoencoder 编码
            
            # 处理 _input_img_tensor，得到 image_latent
            if isinstance(_input_img_tensor, torch.Tensor) and _input_img_tensor.device != device:
                _input_img_tensor = _input_img_tensor.to(device, non_blocking=True)
            if _input_img_tensor.dim() == 3:
                _input_img_tensor = _input_img_tensor.unsqueeze(0)
            
            input_moments_256 = autoencoder(_input_img_tensor, fn='encode_moments')
            input_moments_256 = input_moments_256.detach()
            _input_image_latent = autoencoder.sample(input_moments_256)
            
            _context, _token_mask = utils.get_input_image_embeddings_and_masks(
                batch_input_images=_input_img,  # PIL Image 列表或路径列表，自动检测
                vl_chat_processor=vl_chat_processor,
                vl_gpt=vl_gpt,
                device=device,
                question="",
                num_image_tokens=576,
                output_tokens=576,  # 返回前77个token: [batch_size, 77, 2048], [batch_size, 77]
                accelerator=accelerator,
                cached_input_ids=cached_training_input_ids  # 使用预编码的input_ids，优化性能
            )
            

            assert _context.size(0) == _n_samples
            assert not return_caption # during training we should not use this 
            if return_caption:
                raise Exception("return_caption = True!")
                
            elif return_clipScore:
                raise Exception("return_clipScore = True!")
                
            else:
                # 生成图片，传入 image_latent
                generated_samples = ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask, image_latent=_input_image_latent)
                # 确保 _output_img 在正确的设备上
                if isinstance(_output_img, torch.Tensor) and _output_img.device != device:
                    _output_img = _output_img.to(device, non_blocking=True)
                # 确保 _output_img 有正确的 batch 维度
                # 当 batch_size=1 时，wds.batched 可能返回 [C, H, W] 而不是 [1, C, H, W]
                if isinstance(_output_img, torch.Tensor) and _output_img.dim() == 3:
                    _output_img = _output_img.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
                # 返回生成的图片和对应的ground truth图片
                return generated_samples, _output_img
        
        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            # 使用新的保存函数，同时保存生成的图片和ground truth图片（拼接在一起）
            utils.sample2dir_with_gt(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, return_clipScore=False, ClipSocre_model=ClipSocre_model, config=config)
            _fid = 0
            if accelerator.is_main_process: # 主进程（主卡）会计算FID
                inception_ckpt_path = getattr(config, 'inception_ckpt_path', None)
                _fid = calculate_fid_given_paths((dataset.fid_stat, path), inception_ckpt_path=inception_ckpt_path)

                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
                # 读取保存的图片并单独上传到 wandb
                # sample2dir_with_gt 保存的图片是水平拼接的 [generated, gt]
                # 读取所有保存的图片并分别上传，key 中包含 step 信息以便区分不同时间点的评估结果
                eval_images = []
                for i in range(n_samples):
                    img_path = os.path.join(path, f"{i}.png")
                    if os.path.exists(img_path):
                        # 读取图片
                        img_pil = Image.open(img_path).convert("RGB")
                        eval_images.append(wandb.Image(img_pil, caption=f"eval_sample_{i}_step_{train_state.step}"))
                
                if eval_images:
                    # 使用列表方式上传多张图片，key 中包含 step 信息
                    wandb.log({f'eval_samples_{n_samples}_step_{train_state.step}': eval_images}, step=train_state.step)
                    logging.info(f'Uploaded {len(eval_images)} evaluation samples to wandb at step {train_state.step}')
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum') # 用 accelerator.reduce 把主进程的FID广播/规约到所有进程

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = next(data_generator)
        # 确保batch中的tensor在正确的设备上（accelerator.prepare应该已经处理了，但确保万无一失）
        # 注意：_batch[0] 是路径列表（字符串）或tensor，_batch[1] 是output图像tensor，_batch[2] 是input图像tensor
        if isinstance(batch[1], torch.Tensor) and batch[1].device != device:
            batch = (batch[0], batch[1].to(device, non_blocking=True), batch[2].to(device, non_blocking=True) if isinstance(batch[2], torch.Tensor) else batch[2])
        elif len(batch) == 3 and isinstance(batch[2], torch.Tensor) and batch[2].device != device:
            batch = (batch[0], batch[1], batch[2].to(device, non_blocking=True))
        metrics = train_step(batch, ss_empty_context)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        ############# save rigid image
        if train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            if hasattr(dataset, "vis_image_paths"):
                # 优化：只加载需要的图像（前n_samples_eval个），而不是全部加载
                vis_image_paths = dataset.vis_image_paths[:config.train.n_samples_eval]
                
                # 加载 input images 并编码得到 image_latent
                from datasets import center_crop_arr
                input_images_tensor = []
                for img_path in vis_image_paths:
                    input_pil = Image.open(img_path).convert("RGB")
                    input_arr = center_crop_arr(input_pil, image_size=config.dataset.resolution)
                    input_arr = (input_arr / 127.5 - 1.0).astype(np.float32)
                    input_tensor = torch.from_numpy(einops.rearrange(input_arr, 'h w c -> c h w')).to(device)
                    input_images_tensor.append(input_tensor)
                
                input_images_tensor = torch.stack(input_images_tensor, dim=0)  # [n_samples_eval, 3, H, W]
                input_moments_256 = autoencoder(input_images_tensor, fn='encode_moments')
                input_moments_256 = input_moments_256.detach()
                vis_input_image_latent = autoencoder.sample(input_moments_256)  # [n_samples_eval, 4, H, W]
                
                contexts, token_mask = utils.get_input_image_embeddings_and_masks(
                    batch_input_images=vis_image_paths,
                    vl_chat_processor=vl_chat_processor,
                    vl_gpt=vl_gpt,
                    device=device,
                    question="",
                    num_image_tokens=576,
                    output_tokens=576,  # 返回前77个token: [batch_size, 77, 2048], [batch_size, 77]
                    accelerator=accelerator,
                    cached_input_ids=cached_training_input_ids  # 使用预编码的input_ids，优化性能
                )
                _context = contexts  # 已经只加载了需要的数量
                _token_mask = token_mask
            else:
                raise NotImplementedError()
            samples = ode_fm_solver_sample(nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50, context=_context, token_mask=_token_mask, image_latent=vis_input_image_latent)
            samples_unpreprocessed = dataset.unpreprocess(samples)  # [batch_size, 3, H, W], 范围 [0, 1]
            
            if accelerator.is_main_process:
                target_size = 256
                
                # 加载 input 图片（用于显示）
                input_images_pil = dataset.get_vis_images_as_pil(max_images=config.train.n_samples_eval)
                
                # 加载 output 图片（ground truth）
                if hasattr(dataset, "get_vis_output_images_as_pil"):
                    gt_images_pil = dataset.get_vis_output_images_as_pil(max_images=config.train.n_samples_eval)
                else:
                    gt_images_pil = []
                
                # 确定设备：使用 samples_unpreprocessed 的设备，或者 accelerator.device
                if len(samples_unpreprocessed) > 0:
                    target_device = samples_unpreprocessed[0].device
                else:
                    target_device = accelerator.device
                
                # 准备拼接的图片列表：每行3张图片 [input, ground truth, generated]
                all_images_for_grid = []
                
                for i in range(min(config.train.n_samples_eval, len(samples_unpreprocessed))):
                    # 第1张：input image
                    if i < len(input_images_pil):
                        input_pil = input_images_pil[i]
                        # 转换为tensor并resize到256x256
                        input_tensor = ToTensor()(input_pil).to(target_device)  # [C, H, W], 范围 [0, 1]，移到GPU
                        if input_tensor.shape[1] != target_size or input_tensor.shape[2] != target_size:
                            input_tensor = input_tensor.unsqueeze(0)  # [1, C, H, W]
                            input_tensor = F.interpolate(input_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
                            input_tensor = input_tensor.squeeze(0)  # [C, H, W]
                        all_images_for_grid.append(input_tensor)
                    else:
                        # 占位图像（也在GPU上创建）
                        all_images_for_grid.append(torch.zeros((3, target_size, target_size), device=target_device))
                    
                    # 第2张：ground truth image
                    if i < len(gt_images_pil):
                        gt_pil = gt_images_pil[i]
                        gt_tensor = ToTensor()(gt_pil).to(target_device)  # [C, H, W], 范围 [0, 1]，移到GPU
                        if gt_tensor.shape[1] != target_size or gt_tensor.shape[2] != target_size:
                            gt_tensor = gt_tensor.unsqueeze(0)  # [1, C, H, W]
                            gt_tensor = F.interpolate(gt_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
                            gt_tensor = gt_tensor.squeeze(0)  # [C, H, W]
                        all_images_for_grid.append(gt_tensor)
                    else:
                        # 占位图像（也在GPU上创建）
                        all_images_for_grid.append(torch.zeros((3, target_size, target_size), device=target_device))
                    
                    # 第3张：generated image
                    generated = samples_unpreprocessed[i]  # [C, H, W], 范围 [0, 1]
                    # 确保在正确的设备上
                    if generated.device != target_device:
                        generated = generated.to(target_device)
                    if generated.shape[1] != target_size or generated.shape[2] != target_size:
                        generated = generated.unsqueeze(0)  # [1, C, H, W]
                        generated = F.interpolate(generated, size=(target_size, target_size), mode='bilinear', align_corners=False)
                        generated = generated.squeeze(0)  # [C, H, W]
                    all_images_for_grid.append(generated)
                
                # 使用 make_grid 创建网格，每行3张图片，添加 padding 使图片更好看
                if all_images_for_grid:
                    # 将所有图片堆叠成 [n_images, C, H, W] 形状（都在GPU上）
                    images_tensor = torch.stack(all_images_for_grid, dim=0)  # [n_samples_eval*3, C, H, W]
                    # 使用 make_grid，nrow=3 表示每行3张图片，padding=2 表示图片间距为2像素
                    final_image = make_grid(images_tensor, nrow=3, padding=2, pad_value=1.0)
                    save_image(final_image, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                    wandb.log({'samples': wandb.Image(final_image)}, step=train_state.step)
                else:
                    # 如果没有图片，使用原来的逻辑作为fallback
                    samples = make_grid(samples_unpreprocessed, 5)
                    save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
                    wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

         ############ save checkpoint and evaluate results
        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            
            # 加强进程同步,所有进程先等待
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                import shutil
                import time
                
                ckpt_path = os.path.join(config.ckpt_root, f'{train_state.step}.ckpt')
                
                # 清理整个checkpoint目录中过期的临时文件
                try:
                    for item in os.listdir(config.ckpt_root):
                        if '.tmp_' in item or item.endswith('.backup'):
                            tmp_path = os.path.join(config.ckpt_root, item)
                            try:
                                if os.path.isdir(tmp_path):
                                    # 检查修改时间，如果超过1小时则删除
                                    if time.time() - os.path.getmtime(tmp_path) > 3600:
                                        logging.info(f'Removing stale temporary directory: {tmp_path}')
                                        shutil.rmtree(tmp_path)
                            except Exception as e:
                                logging.warning(f'Error cleaning temporary file {tmp_path}: {e}')
                except Exception as e:
                    logging.warning(f'Error scanning checkpoint directory: {e}')
            
            # 同步，确保临时文件清理完成
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                ckpt_path = os.path.join(config.ckpt_root, f'{train_state.step}.ckpt')
                try:
                    train_state.save(ckpt_path)
                except Exception as e:
                    logging.error(f'Failed to save checkpoint at step {train_state.step}: {e}')
                    # 不要让保存失败中断训练
                    logging.warning('Continuing training despite checkpoint save failure')
            
            accelerator.wait_for_everyone()
            ######需要修改n_samples
            fid = eval_step(n_samples=50, sample_steps=50)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))

            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    ##############需要修改###############
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("workdir_base", None, "Base directory for workdir. If not provided, uses default path.")
flags.DEFINE_string("vae_pretrained_path", None, "Path to pretrained VAE checkpoint.")
flags.DEFINE_string("model_pretrained_path", None, "Path to pretrained model checkpoint.")
flags.DEFINE_string("fid_stat_path", None, "Path to FID statistics file.")
flags.DEFINE_string("inception_ckpt_path", None, "Path to Inception checkpoint.")
flags.DEFINE_string("sample_path", None, "Path to save samples.")
flags.DEFINE_string("train_tar_pattern", None, "Training tar pattern for WebDataset.")
flags.DEFINE_string("test_tar_pattern", None, "Test tar pattern for WebDataset.")
flags.DEFINE_string("vis_image_root", None, "Path to visualization images root.")
flags.DEFINE_string("resume_ckpt_root", None, "Path to checkpoint root directory for resuming. If not provided, uses workdir/ckpts.")

# WandB parameters
flags.DEFINE_string("wandb_project", None, "WandB project name. If not provided, uses config.wandb_project or default naming.")
flags.DEFINE_enum("wandb_mode", None, ["online", "offline", "disabled"], "WandB mode: online (sync to cloud), offline (local only), or disabled.")

# Training parameters
flags.DEFINE_integer("n_steps", None, "Total training iterations.")
flags.DEFINE_integer("batch_size", None, "Overall batch size across ALL gpus.")
flags.DEFINE_integer("log_interval", None, "Iteration interval for logging.")
flags.DEFINE_integer("eval_interval", None, "Iteration interval for visual testing.")
flags.DEFINE_integer("save_interval", None, "Iteration interval for saving checkpoints.")
flags.DEFINE_integer("n_samples_eval", None, "Number of samples for evaluation.")

# Dataset parameters
flags.DEFINE_string("dataset_name", None, "Dataset name.")
flags.DEFINE_string("task", None, "Task name.")
flags.DEFINE_integer("resolution", None, "Dataset resolution.")
flags.DEFINE_integer("shuffle_buffer", None, "Shuffle buffer size for WebDataset.")
flags.DEFINE_boolean("resampled", None, "Whether to resample WebDataset.")
flags.DEFINE_boolean("split_data_by_node", None, "Whether to split data by node.")
flags.DEFINE_integer("estimated_samples_per_shard", None, "Estimated samples per shard.")
flags.DEFINE_string("sampling_weights", None, "Sampling weights for multiple tar patterns (format: '0.7,0.3').")

# Sample parameters
flags.DEFINE_integer("sample_steps", None, "Sample steps during inference/testing.")
flags.DEFINE_integer("n_samples", None, "Number of samples for testing.")
flags.DEFINE_integer("mini_batch_size", None, "Batch size for testing.")
flags.DEFINE_integer("scale", None, "CFG scale.")

# Optimizer parameters
flags.DEFINE_string("optimizer_name", None, "Optimizer name.")
flags.DEFINE_float("lr", None, "Learning rate.")
flags.DEFINE_float("weight_decay", None, "Weight decay.")
flags.DEFINE_string("betas", None, "Betas for optimizer (format: '0.9,0.9').")
flags.DEFINE_enum("adamw_impl", None, ["torch", "bitsandbytes", "AdamW", "AdamW8bit"], "Select AdamW backend.")

# DataLoader parameters
flags.DEFINE_integer("num_workers", None, "Number of workers for DataLoader.")

# Model parameters
flags.DEFINE_boolean("use_cross_attention", None, "Whether to use cross attention in the first stage config.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    
    # 设置 workdir
    if FLAGS.workdir:
        # 如果直接提供了 workdir，直接使用
        config.workdir = FLAGS.workdir
    else:
        # 否则根据 workdir_base 或默认路径构建
        default_workdir_base = '/storage/v-jinpewang/lab_folder/junchao/Crossflow_training/Crossflow_2d_wd/work/workdir_wo_textbox'
        workdir_base = FLAGS.workdir_base or default_workdir_base
        config.workdir = os.path.join(workdir_base, config.config_name, config.hparams)
    # ckpt_root 始终用于保存 checkpoint，始终使用 workdir/ckpts
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    # resume_ckpt_root 用于 resume，如果指定了则使用指定的路径，否则使用 ckpt_root
    if FLAGS.resume_ckpt_root:
        config.resume_ckpt_root = FLAGS.resume_ckpt_root
    else:
        config.resume_ckpt_root = config.ckpt_root
    config.sample_dir = os.path.join(config.workdir, 'samples')

    # WandB 配置：命令行参数优先级最高
    if FLAGS.wandb_project:
        config.wandb_project = FLAGS.wandb_project
    if FLAGS.wandb_mode:
        config.wandb_mode = FLAGS.wandb_mode
    
    # 如果通过命令行传入路径，则覆盖配置文件中的路径
    if FLAGS.vae_pretrained_path:
        config.autoencoder.pretrained_path = FLAGS.vae_pretrained_path
    if FLAGS.model_pretrained_path:
        config.pretrained_path = FLAGS.model_pretrained_path
    if FLAGS.fid_stat_path:
        config.fid_stat_path = FLAGS.fid_stat_path
    if FLAGS.inception_ckpt_path:
        config.inception_ckpt_path = FLAGS.inception_ckpt_path
    if FLAGS.sample_path:
        config.sample.path = FLAGS.sample_path
    if FLAGS.train_tar_pattern:
        config.dataset.train_tar_pattern = FLAGS.train_tar_pattern
    if FLAGS.test_tar_pattern:
        config.dataset.test_tar_pattern = FLAGS.test_tar_pattern
    if FLAGS.vis_image_root:
        config.dataset.vis_image_root = FLAGS.vis_image_root
    
    # Training parameters
    if FLAGS.n_steps is not None:
        config.train.n_steps = FLAGS.n_steps
    if FLAGS.batch_size is not None:
        config.train.batch_size = FLAGS.batch_size
    if FLAGS.log_interval is not None:
        config.train.log_interval = FLAGS.log_interval
    if FLAGS.eval_interval is not None:
        config.train.eval_interval = FLAGS.eval_interval
    if FLAGS.save_interval is not None:
        config.train.save_interval = FLAGS.save_interval
    if FLAGS.n_samples_eval is not None:
        config.train.n_samples_eval = FLAGS.n_samples_eval
    
    # Dataset parameters
    if FLAGS.dataset_name is not None:
        config.dataset.name = FLAGS.dataset_name
    if FLAGS.task is not None:
        config.dataset.task = FLAGS.task
    if FLAGS.resolution is not None:
        config.dataset.resolution = FLAGS.resolution
    if FLAGS.shuffle_buffer is not None:
        config.dataset.shuffle_buffer = FLAGS.shuffle_buffer
    if FLAGS.resampled is not None:
        config.dataset.resampled = FLAGS.resampled
    if FLAGS.split_data_by_node is not None:
        config.dataset.split_data_by_node = FLAGS.split_data_by_node
    if FLAGS.estimated_samples_per_shard is not None:
        config.dataset.estimated_samples_per_shard = FLAGS.estimated_samples_per_shard
    if FLAGS.sampling_weights is not None:
        # Parse sampling_weights string like "0.7,0.3" to list [0.7, 0.3]
        sampling_weights_values = [float(x.strip()) for x in FLAGS.sampling_weights.split(',')]
        config.dataset.sampling_weights = sampling_weights_values
    
    # Sample parameters
    if FLAGS.sample_steps is not None:
        config.sample.sample_steps = FLAGS.sample_steps
    if FLAGS.n_samples is not None:
        config.sample.n_samples = FLAGS.n_samples
    if FLAGS.mini_batch_size is not None:
        config.sample.mini_batch_size = FLAGS.mini_batch_size
    if FLAGS.scale is not None:
        config.sample.scale = FLAGS.scale
    
    # Optimizer parameters
    if FLAGS.optimizer_name is not None:
        config.optimizer.name = FLAGS.optimizer_name
    if FLAGS.lr is not None:
        config.optimizer.lr = FLAGS.lr
    if FLAGS.weight_decay is not None:
        config.optimizer.weight_decay = FLAGS.weight_decay
    if FLAGS.betas is not None:
        # Parse betas string like "0.9,0.9" to tuple (0.9, 0.9)
        betas_values = [float(x.strip()) for x in FLAGS.betas.split(',')]
        config.optimizer.betas = tuple(betas_values)
    if FLAGS.adamw_impl is not None:
        config.optimizer.adamw_impl = FLAGS.adamw_impl
    
    # DataLoader parameters
    if FLAGS.num_workers is not None:
        config.num_workers = FLAGS.num_workers
    
    # Model parameters
    if FLAGS.use_cross_attention is not None:
        # 设置第一个 stage_config 的 use_cross_attention
        if hasattr(config.nnet.model_args, 'stage_configs') and len(config.nnet.model_args.stage_configs) > 0:
            config.nnet.model_args.stage_configs[0].use_cross_attention = FLAGS.use_cross_attention
    
    train(config)


if __name__ == "__main__":
    app.run(main)