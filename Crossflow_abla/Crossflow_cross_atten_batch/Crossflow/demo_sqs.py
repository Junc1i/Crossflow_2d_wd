# train_t2i_modified.py
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
import os
import wandb
import numpy as np
import time
import random

import libs.autoencoder
from libs.t5 import T5Embedder
from libs.clip import FrozenCLIPEmbedder
from diffusion.flow_matching import FlowMatching, ODEFlowMatchingSolver, ODEEulerFlowMatchingSolver
from tools.fid_score import calculate_fid_given_paths
from tools.clip_score import ClipSocre
# ...（保留原有import）
from spae import SPAE  # 导入之前实现的SPAE

def train(config):
    # ...（保留原有初始化部分）
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
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

    dataset = get_dataset(**config.dataset)
    print("FID stat path:", dataset.fid_stat)
    assert os.path.exists(dataset.fid_stat)

    gpu_model = torch.cuda.get_device_name(torch.cuda.current_device())
    num_workers = 8

    train_dataset = dataset.get_split(split='train', labeled=True)
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                    num_workers=num_workers, pin_memory=True, persistent_workers=True)

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, drop_last=True,
                                     num_workers=num_workers, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    # 新增：初始化SPAE模型
    spae = SPAE(llm_name=config.spae.llm_name, 
                      clip_model=config.spae.clip_model)
    spae = accelerator.prepare(spae)
    spae.to(device)
    
    # SPAE单独训练阶段
    if config.train_spae:
        logging.info("Start training SPAE...")
        spae_optimizer = torch.optim.Adam(spae.encoder.parameters(), lr=1e-4)
        for epoch in range(config.spae_epochs):
            for batch in train_dataset_loader:
                images = batch[0].to(device)  # 假设第一个元素是图像
                outputs = spae(images)
                loss = outputs["semantic_loss"]
                
                spae_optimizer.zero_grad()
                accelerator.backward(loss)
                spae_optimizer.step()
            logging.info(f"SPAE Epoch {epoch} Loss: {loss.item()}")
        torch.save(spae.state_dict(), f"{config.ckpt_root}/spae_final.pt")
        return # 结束SPAE单独训练

    # 加载预训练SPAE
    spae.load_state_dict(torch.load(f"{config.ckpt_root}/spae_final.pt")) 
    spae.eval()

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    ss_empty_context = None

    ClipSocre_model = ClipSocre(device=device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator(autoencoder):
        while True:
            for data in test_dataset_loader:
                if len(data) == 5:
                    _img, _context, _token_mask, _token, _caption = data
                else:
                    _img, _context = data
                    _token_mask = None
                    _token = None
                    _caption = None
                
                if len(_img.shape)==5:
                    _testbatch_img_blurred = autoencoder.sample(_img[:,1,:]) 
                    yield _context, _token_mask, _token, _caption, _testbatch_img_blurred
                else:
                    assert len(_img.shape)==4
                    yield _context, _token_mask, _token, _caption, None

    context_generator = get_context_generator(autoencoder)

    _flow_mathcing_model = FlowMatching()

    # 修改训练步骤
    def train_step(_batch, _ss_empty_context):
        _metrics = dict()
        optimizer.zero_grad()

        assert len(_batch)==6
        assert not config.dataset.cfg
        _batch_img = _batch[0]
        _batch_con = _batch[1]
        _batch_mask = _batch[2]
        _batch_token = _batch[3]
        _batch_caption = _batch[4]
        _batch_img_ori = _batch[5]

        _z = autoencoder.sample(_batch_gen_img) 
        _z_cond = spae(_batch_input_img) #生成语义token
        
        # 保留原有处理流程
        loss, loss_dict = _flow_mathcing_model(_z, nnet, loss_coeffs=config.loss_coeffs, cond=_z_cond, con_mask=_batch_mask, batch_img_clip=_batch_img_ori, \
            nnet_style=config.nnet.name, text_token=_batch_token, model_config=config.nnet.model_args, all_config=config, training_step=train_state.step)

        
        # ...（保留原有优化步骤）
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


    # ...（保留其余训练循环和评估代码）

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=ODE_Euler_Flow_Matching_Solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')
        
        def sample_fn(_n_samples, return_caption=False, return_clipScore=False, ClipSocre_model=None, config=None):
            _context, _token_mask, _token, _caption, _testbatch_img_blurred = next(context_generator)
            assert _context.size(0) == _n_samples
            assert not return_caption # during training we should not use this 
            if return_caption:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask), _caption
            elif return_clipScore:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, caption=_caption)
            else:
                return ode_fm_solver_sample(nnet_ema, _n_samples, sample_steps, context=_context, token_mask=_token_mask)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            clip_score_list = utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess, return_clipScore=True, ClipSocre_model=ClipSocre_model, config=config)
            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                _clip_score_list = torch.cat(clip_score_list)
                logging.info(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid} clip_score{len(_clip_score_list)} = {_clip_score_list.mean().item()}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x, next(data_generator))
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
            if hasattr(dataset, "token_embedding"):
                contexts = torch.tensor(dataset.token_embedding, device=device)[ : config.train.n_samples_eval]
                token_mask = torch.tensor(dataset.token_mask, device=device)[ : config.train.n_samples_eval]
            elif hasattr(dataset, "contexts"):
                contexts = torch.tensor(dataset.contexts, device=device)[ : config.train.n_samples_eval]
                token_mask = None
            else:
                raise NotImplementedError
            samples = ode_fm_solver_sample(nnet_ema, _n_samples=config.train.n_samples_eval, _sample_steps=50, context=contexts, token_mask=token_mask)
            samples = make_grid(dataset.unpreprocess(samples), 5)
            if accelerator.is_main_process:
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
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


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
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)

# 配置文件需要新增SPAE相关参数
spae_config = {
    'llm_name': 'bert-base-uncased',
    'clip_model': 'ViT-B/32',
    'train_spae': True,  # 第一阶段设为True训练SPAE
    'spae_epochs': 20,
}

# 运行时需分两个阶段：
# 第一阶段：train_spae=True 训练SPAE
# 第二阶段：train_spae=False 训练autoencoder+diffusion