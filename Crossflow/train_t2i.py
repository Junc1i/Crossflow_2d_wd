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
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    # 定义 num_workers（可以从配置中读取，或使用默认值）
    gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
    num_workers = 40
    
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
            num_workers=num_workers,  # 可以使用多个 worker 加速数据加载
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
    train_state.resume(config.ckpt_root)

    
#    for param in nnet.parameters():
#        print("para.device:",param.device) 

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, use_safetensors=True
    )
    vl_gpt = vl_gpt.half().eval().to(device)

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

        assert len(_batch)==2
        assert not config.dataset.cfg
        # WebDataset 模式：_batch[0] 是 PIL Image 列表
        # 文件系统模式：_batch[0] 是图像路径列表（字符串列表）
        _batch_input_img = _batch[0]  
        _batch_output_img = _batch[1] # output image tensor
        
        # 确保输出图像在GPU上（DataLoader可能返回CPU tensor）
        if isinstance(_batch_output_img, torch.Tensor) and _batch_output_img.device != device:
            _batch_output_img = _batch_output_img.to(device, non_blocking=True)
        
        moments_256 = autoencoder(_batch_output_img, fn='encode_moments')
        # moments_256 应该是 [batch_size, 8, H, W] 形状，需要保持batch维度
        moments_256 = moments_256.detach()  # 保持为 Tensor，不要转换为 numpy

        _z = autoencoder.sample(moments_256) # 传入moments，返回 [batch_size, 4, H, W]
        
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
            output_tokens=77,  # 返回前77个token: [batch_size, 77, 2048], [batch_size, 77]
            accelerator=accelerator,
            cached_input_ids=cached_training_input_ids  # 传入预编码的input_ids，跳过tokenizer.encode
        )
        with accelerator.accumulate(nnet):    
            loss, loss_dict = _flow_mathcing_model(_z, nnet, loss_coeffs=config.loss_coeffs, cond=_batch_con, con_mask=_batch_mask, batch_img_clip=_batch_output_img, \
            nnet_style=config.nnet.name, text_token=None, model_config=config.nnet.model_args, all_config=config, training_step=train_state.step)


            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            for key in loss_dict.keys():
                _metrics[key] = accelerator.gather(loss_dict[key].detach()).mean()
            accelerator.backward(loss.mean())
            optimizer.step()
            lr_scheduler.step()
            train_state.ema_update(config.get('ema_rate', 0.9999))
            train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, context=None, caption=None, testbatch_img_blurred=None, two_stage_generation=-1, token_mask=None, return_clipScore=False, ClipSocre_model=None):
        with torch.no_grad():
            _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)
                
            _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = _z_gaussian.shape, mask=token_mask)
            _z_init = _z_x0.reshape(_z_gaussian.shape)
            
            assert config.sample.scale > 1
            _cfg = config.sample.scale

            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")

            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, step_size_type="step_in_dsigma", guidance_scale=_cfg)
            _z, _ = ode_solver.sample(x_T=_z_init, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator)

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
            _input_img, _output_img = next(context_generator) 
            # _input_img 可能是 PIL Image 列表（WebDataset）或路径列表（文件系统）
            # _output_img 是 output 图像 tensor，形状为 [batch_size, 3, H, W] 或 [batch_size, C, H, W]
            _context, _token_mask = utils.get_input_image_embeddings_and_masks(
                batch_input_images=_input_img,  # PIL Image 列表或路径列表，自动检测
                vl_chat_processor=vl_chat_processor,
                vl_gpt=vl_gpt,
                device=device,
                question="",
                num_image_tokens=576,
                output_tokens=77,  # 返回前77个token: [batch_size, 77, 2048], [batch_size, 77]
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
                # 生成图片
                generated_samples = ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask)
                # 确保 _output_img 在正确的设备上
                if isinstance(_output_img, torch.Tensor) and _output_img.device != device:
                    _output_img = _output_img.to(device, non_blocking=True)
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
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum') # 用 accelerator.reduce 把主进程的FID广播/规约到所有进程

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = next(data_generator)
        # 确保batch中的tensor在正确的设备上（accelerator.prepare应该已经处理了，但确保万无一失）
        # 注意：_batch[0] 是路径列表（字符串），_batch[1] 是图像tensor
        if isinstance(batch[1], torch.Tensor) and batch[1].device != device:
            batch = (batch[0], batch[1].to(device, non_blocking=True))
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
                
                contexts, token_mask = utils.get_input_image_embeddings_and_masks(
                    batch_input_images=vis_image_paths,
                    vl_chat_processor=vl_chat_processor,
                    vl_gpt=vl_gpt,
                    device=device,
                    question="",
                    num_image_tokens=576,
                    output_tokens=77,  # 返回前77个token: [batch_size, 77, 2048], [batch_size, 77]
                    accelerator=accelerator,
                    cached_input_ids=cached_training_input_ids  # 使用预编码的input_ids，优化性能
                )
                _context = contexts  # 已经只加载了需要的数量
                _token_mask = token_mask
            else:
                raise NotImplementedError()
            samples = ode_fm_solver_sample(nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50, context=_context, token_mask=_token_mask)
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

            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            ######需要修改n_samples
            fid = eval_step(n_samples=500, sample_steps=50)  # calculate fid of the saved checkpoint
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
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    
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
    
    train(config)


if __name__ == "__main__":
    app.run(main)