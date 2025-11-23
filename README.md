# WebDataset版本
## 优化
### Data
将对应的input和output image组织成一对样本存入shard文件，每个shard存放10k pairs。
使用WebDataset读取数据，每次只把需要的样本加载进内存。
### GPU utilization
优化tokenizer.encode每次需要在cpu上转化的问题，GPU利用率不会为0。
但是由于直接加载图片再送入visual encoder处理，每次为了用满显存拿的batch是相对于trainable model，而这个batch对于visual encoder来说用不满显存，所以有时候GPU利用率会是50%。

## 训练配置
### 环境配置
```bash
git clone git@github.com:Junc1i/Crossflow_2d_wd.git
conda create -n crossflow python=3.10 -y
conda activate crossflow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install Cython
pip install bitsandbytes
cd Crossflow
pip install -r requirements.txt
cd Janus
pip install -e .

pip install webdataset # 新增这一个即可，上面与之前的一致
```

### Prepare training data
使用organize_shard_data.py组织的shard文件

### Model Preparation
下载预训练权重
https://huggingface.co/QHL067/CrossFlow/blob/main/pretrained_models/t2i_256px_clip_dimr.pth
下载image vae权重
https://huggingface.co/QHL067/CrossFlow/blob/main/assets.tar
使用使用assets/stable-diffusion/autoencoder_kl.pth
下载fid_stat
下载同上，使用的是assets/fid_stats/fid_stats_mscoco256_val.npz
下载inception_v3 权重
https://huggingface.co/zgcr654321/pretrained_models/blob/main/inception_v3_pytorch_weights/pt_inception-2015-12-05-6726825d.pth

### Training
config.train.n_steps（这个需要算一下，epoch = n_steps * batch_size / n_steps)
**config.train.batch_size（不爆显存的情况下尽量往大开，但必须保证batch_size / number_of_gpus是整数，八卡就是8的倍数）**
**train_t2i.py中第95行num_workers需要调整一下，现在是num_workers=40.wds.WebLoader不支持prefetch_factor**
关于optimizer
Crossflow/utils.py中第80行，先使用
```python
elif name == 'adamw':
#        from torch.optim import AdamW
#        return AdamW(params, **kwargs)
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(params, **kwargs)
```
运行命令

``` bash
cd Crossflow  
# 单机多卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --main_process_port 16663 \
  --multi_gpu \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision fp16 \
  train_t2i.py \
  --config=configs/t2i_training_demo.py \
  --workdir_base=/your/custom/workdir/base \ # 保存保存模型权重，训练可视化样本以及日志的路径
  --vae_pretrained_path=/path/to/autoencoder_kl.pth \ # Image vae权重路径
  --model_pretrained_path=/path/to/t2i_256px_clip_dimr.pth \ # 预训练权重路径
  --fid_stat_path=/path/to/fid_stats_mscoco256_val.npz \  # fid_stats_mscoco256_val.npz路径
  --inception_ckpt_path=/path/to/pt_inception-2015-12-05-6726825d.pth \ # inceptionv3权重路径
  --sample_path=/path/to/samplesave_wo_textbox \ # 测试集采样保存图片路径
  --train_tar_pattern='/path/to/pairs-{000000..000xxx}.tar' \ # 指定训练集shard文件
  --test_tar_pattern='/path/to/pairs-000xxx.tar' \ # 指定测试集shard文件，3个即可
  --vis_image_root=/path/to/run_vis/ \ # # 使用之前的那15个可视化图片的路径，该路径下需要有input和output两个子文件夹
  2>&1 | tee log.txt
# 多机多卡
accelerate launch \
  --main_process_ip xxxx \ # hostname -I 获取主节点IP
  --main_process_port 16663 \
  --machine_rank 0 \
  --multi_gpu \
  --num_processes 16 \   # 修改为卡数
  --num_machines 2 \   # 修改机器数
  --mixed_precision fp16 \
  train_t2i.py \
  --config=configs/t2i_training_demo.py \
  --workdir_base=/your/custom/workdir/base \
  --vae_pretrained_path=/path/to/autoencoder_kl.pth \
  --model_pretrained_path=/path/to/t2i_256px_clip_dimr.pth \
  --fid_stat_path=/path/to/fid_stats_mscoco256_val.npz \
  --inception_ckpt_path=/path/to/pt_inception-2015-12-05-6726825d.pth \
  --sample_path=/path/to/samplesave_wo_textbox \
  --train_tar_pattern='/path/to/pairs-{000000..000009}.tar' \
  --test_tar_pattern='/path/to/pairs-000010.tar' \
  --vis_image_root=/path/to/run_vis/ \
  2>&1 | tee log.txt
```

