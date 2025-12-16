"""
    This file is used for T2I generation, it also compute the clip similarity between the generated images and the input prompt
"""
from absl import flags
from absl import app
from ml_collections import config_flags
import os

import ml_collections
import torch
from torch import multiprocessing as mp
import torch.nn as nn
import accelerate
import utils
import tempfile
from absl import logging
import builtins
import einops
import math
import numpy as np
import time
from PIL import Image

from diffusion.flow_matching import FlowMatching, ODEFlowMatchingSolver, ODEEulerFlowMatchingSolver
from tools.clip_score import ClipSocre
import libs.autoencoder
from libs.clip import FrozenCLIPEmbedder
from libs.t5 import T5Embedder


def unpreprocess(x):
        x = 0.5 * (x + 1.)
        x.clamp_(0., 1.)
        return x
    

def batch_decode(_z, decode, batch_size=10):
    """
    The VAE decoder requires large GPU memory. To run the interpolation model on GPUs with 24 GB or smaller RAM, you can use this code to reduce memory usage for the VAE. 
    It works by splitting the input tensor into smaller chunks.
    """
    num_samples = _z.size(0)
    decoded_batches = []

    for i in range(0, num_samples, batch_size):
        batch = _z[i:i + batch_size] 
        decoded_batch = decode(batch)
        decoded_batches.append(decoded_batch)

    image_unprocessed = torch.cat(decoded_batches, dim=0)
    return image_unprocessed

def get_caption(llm, text_model, prompt_dict, batch_size):
    
    if batch_size == 3:
        # only addition or only subtraction
        assert len(prompt_dict) == 2
        _batch_con = list(prompt_dict.values()) + [' ']
    elif batch_size == 4:
        # addition and subtraction
        assert len(prompt_dict) == 3
        _batch_con = list(prompt_dict.values()) + [' ']
    elif batch_size >= 5:
        # linear interpolation
        assert len(prompt_dict) == 2
        _batch_con = [prompt_dict['prompt_1']] + [' '] * (batch_size-2) + [prompt_dict['prompt_2']]

    if llm == "clip":
        _latent, _latent_and_others = text_model.encode(_batch_con)   
        _con = _latent_and_others['token_embedding'].detach()
    elif llm == "t5":
        _latent, _latent_and_others = text_model.get_text_embeddings(_batch_con)
        _con = (_latent_and_others['token_embedding'] * 10.0).detach()
    else:
        raise NotImplementedError
    _con_mask = _latent_and_others['token_mask'].detach()
    _batch_token = _latent_and_others['tokens'].detach()
    _batch_caption = _batch_con
    return (_con, _con_mask, _batch_token, _batch_caption)


def evaluate(config):

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
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=config.output_path)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    ##

    if config.nnet.model_args.clip_dim == 4096:
        llm = "t5"
        t5 = T5Embedder(device=device)
    elif config.nnet.model_args.clip_dim == 768:
        llm = "clip"
        clip = FrozenCLIPEmbedder()
        clip.eval()
        clip.to(device)
    else:
        raise NotImplementedError

    
    config = ml_collections.ConfigDict(config)
    
    if config.test_type == 'interpolation':
        prompt_dict = {'prompt_1':config.prompt_1, 'prompt_2':config.prompt_2}
        for key in prompt_dict.keys():
            assert prompt_dict[key] is not None
        config.sample.mini_batch_size = config.num_of_interpolation
        assert config.sample.mini_batch_size >= 5, "for linear interpolation, please sample at least five image"
    elif config.test_type == 'arithmetic':
        prompt_dict = {'prompt_ori':config.prompt_ori, 'prompt_a':config.prompt_a, 'prompt_s':config.prompt_s}
        keys_to_remove = [key for key, value in prompt_dict.items() if value is None]
        for key in keys_to_remove:
            del prompt_dict[key]
        counter = len(prompt_dict)
        assert prompt_dict['prompt_ori'] is not None
        assert counter == 2 or counter == 3
        config.sample.mini_batch_size = counter + 1
    else:
        raise NotImplementedError
    
    config = ml_collections.FrozenConfigDict(config)

    if llm == "clip":
        context_generator = get_caption(llm, clip, prompt_dict=prompt_dict, batch_size=config.sample.mini_batch_size)
    elif llm == "t5":
        context_generator = get_caption(llm, t5, prompt_dict=prompt_dict, batch_size=config.sample.mini_batch_size)
    else:
        raise NotImplementedError

    ##

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    bdv_nnet = None # We don't use Autoguidance
    ClipSocre_model = ClipSocre(device=device) # we also return clip score

    ####### 
    logging.info(config.sample)
    logging.info(f'sample: n_samples={config.sample.n_samples}, mode=t2i, mixed_precision={config.mixed_precision}')

    
    def ode_fm_solver_sample(nnet_ema, _n_samples, _sample_steps, bdv_nnet=bdv_nnet, context=None, caption=None, testbatch_img_blurred=None, two_stage_generation=-1, token=None, token_mask=None, return_clipScore=False, ClipSocre_model=None):
        with torch.no_grad():
            del testbatch_img_blurred
        
            _z_gaussian = torch.randn(_n_samples, *config.z_shape, device=device)

            if 'dimr' in config.nnet.name or 'dit' in config.nnet.name:
                _z_x0, _mu, _log_var = nnet_ema(context, text_encoder = True, shape = _z_gaussian.shape, mask=token_mask)
                _z_init = _z_x0.reshape(_z_gaussian.shape)
            else:
                raise NotImplementedError
            
            if len(_z_init) == 3:
                if config.prompt_a is not None:
                    assert config.prompt_s is None
                    _z_x0_temp = _z_x0[0] + _z_x0[1]
                elif config.prompt_s is not None:
                    assert config.prompt_a is None
                    _z_x0_temp = _z_x0[0] - _z_x0[1]
                else:
                    raise NotImplementedError
                mean = _z_x0_temp.mean()
                std = _z_x0_temp.std()
                _z_x0[2] = (_z_x0_temp - mean) / std
            elif len(_z_init) == 4:
                _z_x0_temp = _z_x0[0] + _z_x0[1] - _z_x0[2]
                mean = _z_x0_temp.mean()
                std = _z_x0_temp.std()
                _z_x0[3] = (_z_x0_temp - mean) / std
            elif len(_z_init) >= 5:
                tensor_a = _z_init[0]
                tensor_b = _z_init[-1]
                num_interpolations = len(_z_init) - 2
                interpolations = [tensor_a + (tensor_b - tensor_a) * (i / (num_interpolations + 1)) for i in range(1, num_interpolations + 1)]
                _z_init = torch.stack([tensor_a] + interpolations + [tensor_b], dim=0)

            assert config.sample.scale > 1
            if config.cfg != -1:
                _cfg = config.cfg
            else:
                _cfg = config.sample.scale

            has_null_indicator = hasattr(config.nnet.model_args, "cfg_indicator")
            
            _sample_steps = config.sample.sample_steps
            
            ode_solver = ODEEulerFlowMatchingSolver(nnet_ema, bdv_model_fn=bdv_nnet, step_size_type="step_in_dsigma", guidance_scale=_cfg)
            _z, _ = ode_solver.sample(x_T=_z_init, batch_size=_n_samples, sample_steps=_sample_steps, unconditional_guidance_scale=_cfg, has_null_indicator=has_null_indicator)

            if config.save_gpu_memory:
                image_unprocessed = batch_decode(_z, decode)
            else:
                image_unprocessed = decode(_z)
            clip_score = ClipSocre_model.calculate_clip_score(caption, image_unprocessed)
            
            return image_unprocessed, clip_score


    def sample_fn(_n_samples, return_caption=False, return_clipScore=False, ClipSocre_model=None, config=None):
        _context, _token_mask, _token, _caption = context_generator
        assert return_clipScore
        assert not return_caption
        return ode_fm_solver_sample(nnet, _n_samples, config.sample.sample_steps, bdv_nnet=bdv_nnet, context=_context, token=_token, token_mask=_token_mask, return_clipScore=return_clipScore, ClipSocre_model=ClipSocre_model, caption=_caption)
        

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.img_save_path or config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f'Samples are saved in {path}')

        clip_score_list = utils.sample2dir_wCLIP(accelerator, path, config.sample.n_samples, config.sample.mini_batch_size, sample_fn, unpreprocess, return_clipScore=True, ClipSocre_model=ClipSocre_model, config=config)
        if clip_score_list is not None:
            _clip_score_list = torch.cat(clip_score_list)
        if accelerator.is_main_process:
            logging.info(f'nnet_path={config.nnet_path}, clip_score{len(_clip_score_list)}={_clip_score_list.mean().item()}')


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)

flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")
flags.DEFINE_float("cfg", -1, 'cfg scale, will use the scale defined in the config file is not assigned')
flags.DEFINE_string("img_save_path", None, "The path to image log.")

flags.DEFINE_string("test_type", None, "The prompt used for generation.")

flags.DEFINE_string("prompt_1", None, "The prompt used for linear interpolation.")
flags.DEFINE_string("prompt_2", None, "The prompt used for linear interpolation.")
flags.DEFINE_integer("num_of_interpolation", -1, 'number of image being samples for linear interpolation')
flags.DEFINE_boolean('save_gpu_memory', False, 'To save VRAM')

flags.DEFINE_string("prompt_ori", None, "The prompt used for arithmetic operations.")
flags.DEFINE_string("prompt_a", None, "The prompt used for arithmetic operations (addition).")
flags.DEFINE_string("prompt_s", None, "The prompt used for arithmetic operations (subtraction).")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.img_save_path = FLAGS.img_save_path
    config.cfg = FLAGS.cfg
    config.test_type = FLAGS.test_type
    config.prompt_1 = FLAGS.prompt_1
    config.prompt_2 = FLAGS.prompt_2
    config.num_of_interpolation = FLAGS.num_of_interpolation
    config.save_gpu_memory = FLAGS.save_gpu_memory
    config.prompt_ori = FLAGS.prompt_ori
    config.prompt_a = FLAGS.prompt_a
    config.prompt_s = FLAGS.prompt_s
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
