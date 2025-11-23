from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets
import torchvision.transforms as transforms
from scipy.signal import convolve2d
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import time
from tqdm import tqdm
import json
import pickle
import io
import cv2

import libs.clip
import bisect
import webdataset as wds
import braceexpand
import logging


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(exn):
        return True
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def pytorch_worker_info(group=None):
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass
    return rank, world_size, worker, num_workers


def is_multi_node_environment():
    """
    检测是否处于多进程(world_size > 1)环境。
    
    注意:webdataset 的 single_node_only 检查的是 world_size(进程数),不是物理节点数。
    只要 world_size > 1(即使是单节点多GPU),webdataset 就会要求显式添加 nodesplitter。
    这是因为 webdataset 的命名有误导性:"multi-node training" 实际上指的是 "multi-process training"。
    
    webdataset 在 resampled=True 且 world_size > 1 时要求显式调用 split_by_node。
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size() > 1:
                return True
    except Exception:
        pass

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    nnodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", "1")))
    if nnodes > 1:
        return True
    return world_size > 1


def split_data_by_node(urls, strategy="interleaved"):
    """
    在多节点间分配 shard,即使数据在本地存储也建议使用,避免重复训练。
    
    Parameters:
    - urls (list): shard URL 列表
    - strategy (str): "chunk", "interleaved", "shuffled_chunk"
    
    Returns:
    - list: 分配给当前节点的 shard 列表
    """
    print('*'*80)
    print("split_data_by_node ing..................")
    gpus_per_node = torch.cuda.device_count()
    rank, world_size, worker, num_workers = pytorch_worker_info()
    print("rank: {}, world_size: {}, worker: {}, num_workers: {}, gpus_per_node: {}".format(
        rank, world_size, worker, num_workers, gpus_per_node))
    
    node_rank = rank // gpus_per_node
    node_world_size = world_size // gpus_per_node

    # 如果 shard 数量少于节点数,所有节点都使用所有 shard
    if len(urls) < node_world_size:
        print(f"Warning: Only {len(urls)} shards but {node_world_size} nodes. "
              f"All nodes will use all shards to avoid empty assignment.")
        print(f"Node {node_rank} has {len(urls)} URLs of {len(urls)} total.")
        print('*'*80)
        return urls

    if strategy == "chunk":
        urls_per_node = math.ceil(len(urls) / node_world_size)
        start_idx = node_rank * urls_per_node
        end_idx = min(start_idx + urls_per_node, len(urls))
        node_urls = urls[start_idx:end_idx]
    elif strategy == "interleaved":
        node_urls = urls[node_rank::node_world_size]
    elif strategy == "shuffled_chunk":
        shuffled_urls = random.sample(urls, len(urls))
        urls_per_node = math.ceil(len(shuffled_urls) / node_world_size)
        start_idx = node_rank * urls_per_node
        end_idx = min(start_idx + urls_per_node, len(urls))
        node_urls = shuffled_urls[start_idx:end_idx]
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    
    print(f"Node {node_rank} has {len(node_urls)} URLs of {len(urls)} total.")
    print('*'*80)
    return node_urls


def get_dataset_size(shards, estimated_sample_per_shard=1000):
    """
    估算数据集大小(用于 __len__)
    根据 shard 数量估算
    """
    shards_list = list(braceexpand.braceexpand(shards))
    num_shards = len(shards_list)
    
    # # 尝试从 num_samples.json 读取
    # if len(shards_list) > 0:
    #     dir_path = os.path.dirname(shards_list[0])
    #     sizes_filename = os.path.join(dir_path, "num_samples.json")
    #     try:
    #         with open(sizes_filename, "r") as fp:
    #             sizes = json.load(fp)
    #         total_size = sum([int(sizes.get(os.path.basename(shard), 0)) for shard in shards_list])
    #         if total_size > 0:
    #             print(f"Loaded dataset size from {sizes_filename}: {total_size} samples, {num_shards} shards")
    #             return total_size, num_shards
    #     except Exception as e:
    #         print(f"Could not load {sizes_filename}: {e}")
    
    # 回退到估算
    total_size = num_shards * estimated_sample_per_shard
    print(f"Estimating dataset size: {total_size} samples ({num_shards} shards * {estimated_sample_per_shard} samples/shard)")
    return total_size, num_shards


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root, need_squeeze=False, full_feature=False, fix_test_order=False):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)
        self.need_squeeze = need_squeeze
        self.full_feature = full_feature
        self.fix_test_order = fix_test_order

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        if self.full_feature:
            z = np.load(os.path.join(self.root, f'{index}.npy'))

            if self.fix_test_order:
                k = self.n_captions[index] - 1
            else:
                k = random.randint(0, self.n_captions[index] - 1)

            test_item = np.load(os.path.join(self.root, f'{index}_{k}.npy'), allow_pickle=True).item()
            token_embedding = test_item['token_embedding']
            token_mask = test_item['token_mask']
            token = test_item['token']
            caption = test_item['promt']
            return z, token_embedding, token_mask, token, caption
        else:
            z = np.load(os.path.join(self.root, f'{index}.npy'))
            k = random.randint(0, self.n_captions[index] - 1)
            c = np.load(os.path.join(self.root, f'{index}_{k}.npy'))
            if self.need_squeeze:
                return z, c.squeeze()
            else:
                return z, c


class JDBFeatureDataset(Dataset):
    def __init__(self, root, resolution, llm):
        super().__init__()
        json_path = os.path.join(root,'img_text_pair.jsonl')
        self.img_root = os.path.join(root,'imgs')
        self.feature_root = os.path.join(root,'features')
        self.resolution = resolution
        self.llm = llm
        self.file_list = []
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.file_list.append(json.loads(line)['img_path'])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data_item = self.file_list[idx]
        # Extract filename and replace extension with .npy for all image formats
        filename = data_item.split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        feature_path = os.path.join(self.feature_root, base_name + '.npy')
        img_path = os.path.join(self.img_root, data_item)

        train_item = np.load(feature_path, allow_pickle=True).item()
        pil_image = Image.open(img_path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")


        z = train_item[f'image_latent_{self.resolution}']
        token_embedding = train_item[f'token_embedding_{self.llm}']
        token_mask = train_item[f'token_mask_{self.llm}']
        token = train_item[f'token_{self.llm}']
        caption = train_item['batch_caption']

        img = center_crop_arr(pil_image, image_size=self.resolution)
        img = (img / 127.5 - 1.0).astype(np.float32)
        img = einops.rearrange(img, 'h w c -> c h w')

        # return z, token_embedding, token_mask, token, caption, 0, img, 0, 0
        return z, token_embedding, token_mask, token, caption, img


class JDBFullFeatures(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, train_path, val_path, resolution, llm, cfg=False, p_uncond=None, fix_test_order=False):
        super().__init__()
        print('Prepare dataset...')
        self.resolution = resolution

        self.train = JDBFeatureDataset(train_path, resolution=resolution, llm=llm)
        self.test = MSCOCOFeatureDataset(os.path.join(val_path, 'val'), full_feature=True, fix_test_order=fix_test_order)
        assert len(self.test) == 40504
        
        print('Prepare dataset ok')

        # self.empty_context = np.load(os.path.join(val_path, 'empty_context.npy'), allow_pickle=True).item()

        assert not cfg

        # text embedding extracted by clip
        self.prompts, self.token_embedding, self.token_mask, self.token = [], [], [], []
        for f in sorted(os.listdir(os.path.join(val_path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            vis_item = np.load(os.path.join(val_path, 'run_vis', f), allow_pickle=True).item()
            self.prompts.append(vis_item['promt'])
            self.token_embedding.append(vis_item['token_embedding'])
            self.token_mask.append(vis_item['token_mask'])
            self.token.append(vis_item['token'])
        self.token_embedding = np.array(self.token_embedding)
        self.token_mask = np.array(self.token_mask)
        self.token = np.array(self.token)

    @property
    def data_shape(self):
        if self.resolution==512:
            return 4, 64, 64
        else:
            return 4, 32, 32

    @property
    def fid_stat(self):
        # 如果通过命令行或配置传入了 fid_stat_path,则使用它；否则使用默认路径
        if self.fid_stat_path:
            return self.fid_stat_path
        return f'/storage/v-jinpewang/lab_folder/qisheng_data/assets/fid_stats/fid_stats_mscoco256_val.npz'

class TextImageDataset(Dataset):
    def __init__(self, root, resolution, llm, feature_dir='features_2D', test_mode=False):
        super().__init__()
        json_path = os.path.join(root,'img_img_pair.jsonl')
        self.img_root = os.path.join(root,'imgs') #render已经在extract里load过,这里不用再load
        self.feature_root = os.path.join(root, feature_dir)
        self.resolution = resolution
        self.llm = llm
        self.test_mode = test_mode
        self.file_list = []
        with open(json_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.file_list.append(json.loads(line)['img_path'])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data_item = self.file_list[idx]
        # Extract filename and replace extension with .npy for all image formats
        filename = data_item.split('/')[-1]
        base_name = os.path.splitext(filename)[0]
        img_feature_path = os.path.join(self.feature_root, base_name + '.npy')
        img_path = os.path.join(self.img_root, data_item)

        train_item = np.load(img_feature_path, allow_pickle=True).item()
        pil_image = Image.open(img_path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")


        # Check if image_latent exists, if not (for test set), use None as placeholder
        latent_key = f'image_latent_{self.resolution}'
        if latent_key in train_item:
            z = train_item[latent_key]
        else:
            # For test set without image_latent, return None (not used during sampling anyway)
            z = None
            
        token_embedding = train_item[f'token_embedding_{self.llm}'][:77]
        #print("train/test token_embedding.shape:",token_embedding.shape,"note: shoule be (77, 2048)")
     #   token_embedding = token_embedding[:,:768]
        #print("token_embedding.shape:",token_embedding.shape,"note: shoule be (77, 768)")
        token_mask = train_item[f'token_mask_{self.llm}'][:77]
        token_mask = torch.tensor(token_mask)
        #token = train_item[f'token_{self.llm}']
        #caption = train_item['batch_caption']

        img = center_crop_arr(pil_image, image_size=self.resolution)
        img = (img / 127.5 - 1.0).astype(np.float32)
        img = einops.rearrange(img, 'h w c -> c h w')

        # return z, token_embedding, token_mask, token, caption, 0, img, 0, 0
        # 测试模式下只返回 token_embedding 和 token_mask,因为 z 和 img 在测试时不需要
        if self.test_mode:
            return token_embedding, token_mask
        else:
            return z, token_embedding, token_mask, img


class TextImageFeatures(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, train_path, val_path, resolution, llm, cfg=False, p_uncond=None, fix_test_order=False, fid_stat_path=None):
        super().__init__()
        print('Prepare dataset...')
        self.resolution = resolution
        self.fid_stat_path = fid_stat_path

        self.train = TextImageDataset(train_path, resolution=resolution, llm=llm, feature_dir='features_2D', test_mode=False)
        self.test = TextImageDataset(val_path, resolution=resolution, llm=llm, feature_dir='features', test_mode=True)
        # assert len(self.test) == 40504
        
        print('Prepare dataset ok')

        # self.empty_context = np.load(os.path.join(val_path, 'empty_context.npy'), allow_pickle=True).item()

        assert not cfg

        # text embedding extracted by clip
        self.prompts, self.token_embedding, self.token_mask, self.token = [], [], [], []
        for f in sorted(os.listdir(os.path.join(val_path, 'run_vis')), key=lambda x: int(x.split('.')[0])):
            vis_item = np.load(os.path.join(val_path, 'run_vis', f), allow_pickle=True).item()
            self.token_embedding.append(vis_item['token_embedding'][:77])
            self.token_mask.append(vis_item['token_mask'][:77])
        self.token_embedding = np.array(self.token_embedding)
        self.token_mask = np.array(self.token_mask)

    @property
    def data_shape(self):
        if self.resolution==512:
            return 4, 64, 64
        else:
            return 4, 32, 32

class OnlineDataset(Dataset):
    """
    在线加载匹配的 input/output 图像对。
    """
    def __init__(self, image_root, resolution=256, task='visual_instruction'):
        super().__init__()
        self.image_root = image_root
        self.resolution = resolution
        self.task = task
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

        print(f"scanning image pairs under: {image_root}")
        input_map = {}
        output_map = {}
        total_input = 0
        total_output = 0

        for root, dirs, files in os.walk(image_root):
            basename = os.path.basename(root)
            if basename not in {'input', 'output'}:
                continue

            parent_rel = os.path.relpath(os.path.dirname(root), image_root)
            if parent_rel == '.':
                parent_rel = ''

            for filename in files:
                if not self._is_valid_image(filename):
                    continue
                # 去掉文件扩展名,以便匹配不同后缀的同名文件(如 image1.jpg 和 image1.png)
                base_name = os.path.splitext(filename)[0]
                # 去除文件名末尾的特殊后缀(_render, _textbox, _edited, _text_only)
                normalized_name = self._normalize_filename(base_name)
                key = os.path.join(parent_rel, normalized_name) if parent_rel else normalized_name
                full_path = os.path.join(root, filename)
                if basename == 'input':
                    input_map[key] = full_path
                    total_input += 1
                else:
                    output_map[key] = full_path
                    total_output += 1

        common_keys = sorted(set(input_map.keys()) & set(output_map.keys()))
        self.paired_samples = [(input_map[key], output_map[key]) for key in common_keys]

        # 找出被跳过的文件(只有input或只有output的)
        input_only_keys = set(input_map.keys()) - set(output_map.keys())
        output_only_keys = set(output_map.keys()) - set(input_map.keys())
        skipped_inputs = len(input_only_keys)
        skipped_outputs = len(output_only_keys)

        print(f"total input images found: {total_input}")
        print(f"total output images found: {total_output}")
        print(f"matched pairs: {len(self.paired_samples)}")
        if skipped_inputs > 0 or skipped_outputs > 0:
            print(f"skipped unmatched files -> input: {skipped_inputs}, output: {skipped_outputs}")
            # 显示一些被跳过的文件示例(最多显示5个)
            if skipped_inputs > 0:
                sample_inputs = sorted(list(input_only_keys))[:5]
                print(f"  example skipped input-only files: {sample_inputs}")
            if skipped_outputs > 0:
                sample_outputs = sorted(list(output_only_keys))[:5]
                print(f"  example skipped output-only files: {sample_outputs}")

    def _is_valid_image(self, filename):
        return any(filename.endswith(ext) for ext in self.valid_extensions)
    
    def _normalize_filename(self, base_name):
        """
        去除文件名末尾的特殊后缀(_render, _textbox, _edited, _text_only)
        只有末尾有这些后缀才去掉
        按长度从长到短排序,优先匹配最长的后缀
        """
        suffixes_to_remove = ['_text_only', '_textbox', '_render', '_edited']  # 按长度从长到短排序
        for suffix in suffixes_to_remove:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break  # 只去掉一个后缀(匹配最长的)
        return base_name

    def __len__(self):
        return len(self.paired_samples)
    
    def __getitem__(self, idx):
        """
        返回匹配的 input/output 图像:
        - input:返回图像路径(字符串),在训练时由utils函数加载
        - output:进行中心裁剪和归一化,返回tensor
        返回格式:(input_path, output_tensor)
        """
        input_path, output_path = self.paired_samples[idx]
        
        try:
            # 加载output图像并处理
            output_pil = Image.open(output_path)
            output_pil.load()
            output_pil = output_pil.convert("RGB")
            output_arr = center_crop_arr(output_pil, image_size=self.resolution)
            output_arr = (output_arr / 127.5 - 1.0).astype(np.float32)
            output_tensor = torch.from_numpy(einops.rearrange(output_arr, 'h w c -> c h w'))

            # 返回input路径和output tensor(DataLoader可以正常处理字符串和tensor)
            return input_path, output_tensor
            
        except Exception as e:
            print(f"Error loading pair (input: {input_path}, output: {output_path}): {e}")
            # 返回占位数据
            placeholder_path = input_path  # 即使出错也返回原始路径
            placeholder_tensor = torch.zeros((3, self.resolution, self.resolution))
            return placeholder_path, placeholder_tensor


def nodesplitter_identity(urls):
    """返回所有URLs,不进行分割,用于替代本地定义的 no_split
    
    这个函数必须定义在模块级别(类外部),以便在 spawn 模式下可以被 pickle。
    """
    return urls


class WebDatasetDataset(IterableDataset):
    """
    使用 webdataset 加载 input/output 图像对。
    迭代器模式,不会加载到内存。
    支持单节点和多节点分布式训练。
    
    使用组合模式而不是继承,这样可以更好地控制 pipeline 的构建,
    并避免 webdataset 链式调用返回新对象导致的问题。
    """
    def __init__(self, tar_pattern, resolution=256, shuffle_buffer=300, 
                 resampled=True, handler=log_and_continue, 
                 estimated_samples_per_shard=1000,
                 split_data_by_node_flag=True,
                 allow_shared_shards=False,  # 新增:是否允许多个进程共享shard
                 vl_chat_processor=None,
                 device=None,
                 num_workers=None,  # 新增:DataLoader 的 num_workers,用于正确计算 with_epoch
                 batch_size=None):  # 新增:batch_size,用于正确计算 with_epoch
        """
        Args:
            tar_pattern: webdataset 的 tar 文件路径模式,支持 braceexpand
                例如: "/local/data/pairs-{000000..000999}.tar"
            resolution: 图像分辨率
            shuffle_buffer: shuffle buffer 大小
            resampled: 是否使用 resampled 模式(用于分布式训练)
            handler: 错误处理函数
            estimated_samples_per_shard: 每个 shard 的估计样本数(用于 __len__)
            split_data_by_node_flag: 是否在多节点间分配 shard
                - True: 多节点训练时使用,即使数据在本地也建议使用,避免重复训练
                - False: 单节点训练时使用
            allow_shared_shards: 是否允许多个进程共享shard(当shard数量少于进程数时)
                - True: 所有进程都访问所有shard,通过resampled和shuffle避免重复
                - False: 按shard级别分配,某些进程可能没有数据(需要empty_check=False)
            vl_chat_processor: VLChatProcessor 实例(可选)
                如果提供,将在数据集层面预处理输入图像,返回 tensor [3, H, W]
            device: torch.device 或字符串(可选)
                如果提供,预处理后的 tensor 将直接移动到该设备(通常是 GPU)
                如果为 None,将自动检测:如果 CUDA 可用则使用当前 CUDA 设备,否则使用 CPU
            num_workers: DataLoader 的 num_workers(可选)
                如果提供,将用于正确计算每个 worker 的 epoch 大小
                如果不提供,将使用总样本数(可能导致每个 worker 处理全部数据)
            batch_size: batch_size(可选)
                如果提供,将用于更精确地计算 epoch 大小
        """
        super().__init__()  # 初始化 IterableDataset
        self.resolution = resolution
        self.handler = handler
        self.vl_chat_processor = vl_chat_processor
        self.num_workers = num_workers if num_workers is not None else 1
        self.batch_size = batch_size

        # 设置设备
        if device is None:
            # 自动检测:如果 CUDA 可用则使用当前 CUDA 设备,否则使用 CPU
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                self.device = torch.device('cpu')
        else:
            # 如果提供了 device,直接使用
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        
        # 展开 URL 列表
        all_urls = list(braceexpand.braceexpand(tar_pattern))
        
        # 注意:如果使用 resampled=True,webdataset 需要显式添加 nodesplitter
        # 为了满足这个要求,我们有两种策略:
        # 1. 如果 split_data_by_node_flag=True:手动分配 URLs,但 webdataset 仍然需要 nodesplitter 标识
        # 2. 如果 split_data_by_node_flag=False:让 webdataset 自动分配(需要添加 split_by_node)
        need_nodesplitter = split_data_by_node_flag or is_multi_node_environment()

        # 如果允许共享shard,所有进程都访问所有shard
        if allow_shared_shards:
            urls = all_urls
            random.shuffle(urls)  # 打乱顺序,配合resampled和shuffle避免重复
            print(f"Shared shards mode: all {len(urls)} shards accessible by all processes")
            # 当允许共享shard时,不需要nodesplitter(所有进程访问相同shard)
            need_nodesplitter = False
        elif split_data_by_node_flag:
            node_urls = split_data_by_node(all_urls, strategy="interleaved")
            if len(node_urls) == 0:
                node_urls = all_urls
            urls = node_urls
            print(f"Split data by node: using {len(urls)} shards for this node (out of {len(all_urls)} total)")
        else:
            urls = all_urls
            random.shuffle(urls)  # 单节点时只需打乱顺序
            print(f"Single node mode: using all {len(urls)} shards")
        
        # 估算数据集大小
        self.num_shards = len(urls)
        if split_data_by_node_flag:
            # 多节点时,获取总数据集大小(用于参考)
            total_num_samples, total_num_shards = get_dataset_size(tar_pattern, estimated_samples_per_shard)
            # 但计算 epoch 大小时,应该使用分配给当前进程的样本数
            # 每个进程处理的样本数 = 当前进程的 shard 数 * 每个 shard 的估计样本数
            self.num_samples = self.num_shards * estimated_samples_per_shard
            self.total_num_samples = total_num_samples  # 保存总样本数用于参考
        else:
            self.num_samples = self.num_shards * estimated_samples_per_shard
            self.total_num_samples = self.num_samples  # 单节点时,总样本数等于当前样本数
        
        # 非 resampled 模式下检查 shard 数量是否足够
        if not resampled:
            rank, world_size, worker, num_workers_info = pytorch_worker_info()
            total_workers_needed = self.num_workers * world_size if world_size > 1 else self.num_workers
            if self.num_shards < total_workers_needed:
                print(f"Warning: Only {self.num_shards} shards but need {total_workers_needed} workers "
                      f"(num_workers={self.num_workers}, world_size={world_size}). "
                      f"Some workers may not have data. Consider using resampled=True or increasing shard count.")
        
        # 计算 epoch 大小
        # 如果提供了 batch_size,with_epoch 的参数应该是每个 worker 的 batch 数
        # 如果没有提供 batch_size,则使用每个 worker 的样本数
        if self.batch_size is not None:
            # 计算每个 worker 的 batch 数(参考 data.py 第405-412行)
            # 使用当前进程的样本数来计算
            num_batches = math.ceil(self.num_samples / self.batch_size)
            if self.num_workers > 1:
                # 每个 worker 处理一部分 batch
                num_worker_batches = math.ceil(num_batches / self.num_workers)
            else:
                num_worker_batches = num_batches
            epoch_size = num_worker_batches  # with_epoch 接收 batch 数
            self.epoch_size = num_worker_batches * self.batch_size  # 保存样本数用于参考
        else:
            # 如果没有 batch_size,使用样本数
            if self.num_workers > 1:
                epoch_size = math.ceil(self.num_samples / self.num_workers)
            else:
                epoch_size = self.num_samples
            self.epoch_size = epoch_size  # 保存用于 __len__
        
        # 创建基础 WebDataset
        if allow_shared_shards:
            # 共享shard模式:所有进程访问所有shard
            base_pipeline = wds.WebDataset(urls, resampled=resampled, handler=handler, nodesplitter=nodesplitter_identity, empty_check=False)
            print(f"Shared shards mode: all processes access all {len(urls)} shards")
        elif need_nodesplitter and hasattr(wds, "split_by_node"):
            # 传入 nodesplitter 参数,让 webdataset 自动处理进程分割
            # empty_check=False: 当 shard 数量少于 worker 数量时,允许某些进程没有数据而不报错
            base_pipeline = wds.WebDataset(urls, resampled=resampled, handler=handler, nodesplitter=wds.split_by_node, empty_check=False)
            print(f"Using nodesplitter=wds.split_by_node for multi-process training (world_size > 1)")
        else:
            # 单进程模式(world_size == 1),使用默认的 single_node_only
            base_pipeline = wds.WebDataset(urls, resampled=resampled, handler=handler, empty_check=False)
        
        # 构建 pipeline 阶段列表
        pipeline_stages = [base_pipeline]
        
        # 在 worker 之间分割数据(如果 num_workers > 1),确保每个 DataLoader worker 处理不同的数据,避免重复训练
        if self.num_workers > 1:
            pipeline_stages.append(wds.split_by_worker)
            print(f"Added wds.split_by_worker for {self.num_workers} workers")
        
        # 添加其他处理阶段
        pipeline_stages.extend([
            wds.shuffle(shuffle_buffer, handler=handler),  # shuffle 样本
            wds.decode("pil", handler=handler),  # 解码为 PIL Image
            wds.to_tuple("in.png", "out.png", handler=handler),  # 提取图像对
            wds.map_tuple(
                self._preprocess_input,
                self._preprocess_output
            ),  # 预处理
        ])
        
        # 如果提供了 batch_size,在 pipeline 中进行批处理
        # 这样可以使用 wds.WebLoader 并且 batch_size=None
        if self.batch_size is not None:
            pipeline_stages.append(wds.batched(self.batch_size, partial=False))
        
        # 组合所有阶段(参考 data.py 第398行)
        self._pipeline = wds.DataPipeline(*pipeline_stages)
        
        # 限制 epoch 大小,根据计算结果动态设置 epoch 大小
        self._pipeline = self._pipeline.with_epoch(epoch_size)
        
        if self.batch_size is not None:
            print(f"WebDataset initialized: {self.num_shards} shards for this node, "
                  f"estimated {self.num_samples} total samples in dataset, "
                  f"epoch size per worker: {epoch_size} batches ({epoch_size * self.batch_size} samples, num_workers={self.num_workers}, batch_size={self.batch_size})")
        else:
            print(f"WebDataset initialized: {self.num_shards} shards for this node, "
                  f"estimated {self.num_samples} total samples in dataset, "
                  f"epoch size per worker: {epoch_size} samples (num_workers={self.num_workers})")
    
    def _preprocess_input(self, pil_image):
        """预处理 input 图像
        
        如果提供了 vl_chat_processor,返回处理后的 tensor [3, H, W](CPU上)
        否则返回 numpy array(向后兼容)
        
        注意:返回 CPU 张量,使用 pin_memory=True 可以加速传输到 GPU
        """
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")
        
        if self.vl_chat_processor is not None:
            # 使用 image_processor 预处理,返回 tensor
            images_outputs = self.vl_chat_processor.image_processor(
                [pil_image], 
                return_tensors="pt"
            )
            # 返回单个图像的 tensor [3, H, W],去掉 batch 维度
            # 保持在 CPU 上,使用 pin_memory 加速传输
            pixel_values = images_outputs.pixel_values.squeeze(0)  # [3, H, W]
            return pixel_values  # 返回 CPU 张量
        else:
            # 向后兼容:返回 numpy array
            return np.array(pil_image, dtype=np.uint8)
    
    def _preprocess_output(self, pil_image):
        """预处理 output 图像,返回 tensor(CPU上)
        
        注意:返回 CPU 张量,使用 pin_memory=True 可以加速传输到 GPU
        图像会被归一化到 [-1, 1] 并转换为 float32
        """
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")
        output_arr = center_crop_arr(pil_image, image_size=self.resolution)
        # 归一化到 [-1, 1] 并转换为 float32(与其他数据集保持一致)
        output_arr = (output_arr / 127.5 - 1.0).astype(np.float32)
        # 从 numpy 创建 tensor,保持在 CPU 上
        tensor = torch.from_numpy(einops.rearrange(output_arr, 'h w c -> c h w'))
        return tensor  # 返回 CPU 张量,使用 pin_memory 加速传输
    
    def __iter__(self):
        """返回迭代器(不加载到内存)"""
        # 使用 pipeline 的迭代器
        return iter(self._pipeline)
    
    # def __len__(self):
    #     """
    #     返回每个周期的样本数(用于 DataLoader 进度条)
    #     注意:这返回的是 epoch_size,不是总数据集大小
    #     因为 WebDataset 使用 with_epoch() 限制每个周期的长度
    #     """
    #     return self.epoch_size
    
    def set_vl_chat_processor(self, vl_chat_processor):
        """设置 vl_chat_processor(用于在创建 processor 后更新数据集)"""
        self.vl_chat_processor = vl_chat_processor
    
    def set_device(self, device):
        """设置设备(用于在创建数据集后更新设备)"""
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device


class OnlineFeatures(DatasetFactory):
    """
    在线加载的数据集工厂类
    
    支持两种模式:
    1. 文件系统模式:使用 OnlineDataset(从 input/output 目录加载)
    2. WebDataset 模式:使用 WebDatasetDataset(从 tar 文件加载,迭代器模式,不加载到内存)
    
    - Dataset只加载原始图像
    - Image VAE编码在训练循环中进行
    """
    def __init__(self, train_image_root=None, vis_image_root=None, train_tar_pattern=None, test_tar_pattern=None,
                 task='visual_instruction', cfg=False, resolution=256, val_split_ratio=0.01,
                 shuffle_buffer=300, resampled=True, split_data_by_node=True,estimated_samples_per_shard=1000,
                 vl_chat_processor=None, device=None, fid_stat_path=None,
                 num_workers=None, batch_size=None):
        """
        Args:
            train_image_root: 文件系统模式的训练数据根目录(包含 input/output 子目录)
            vis_image_root: 可视化图像根目录
            train_tar_pattern: WebDataset 模式的 tar 文件路径模式,支持 braceexpand
                例如: "/local/data/pairs-{000000..000999}.tar"
            test_tar_pattern: WebDataset 模式的 tar 文件路径模式,支持 braceexpand
                例如: "/local/data/pairs-{000000..000999}.tar"
            task: 任务类型
            cfg: 是否使用 classifier-free guidance
            resolution: 图像分辨率
            val_split_ratio: 验证集比例(仅用于文件系统模式)
            shuffle_buffer: WebDataset 的 shuffle buffer 大小
            resampled: WebDataset 是否使用 resampled 模式(用于分布式训练)
            split_data_by_node: WebDataset 是否在多节点间分配 shard
                - True: 多节点训练时使用(推荐,即使数据在本地也建议使用)
                - False: 单节点训练时使用
            estimated_samples_per_shard: 每个 shard 的估计样本数(用于 __len__)
            vl_chat_processor: VLChatProcessor 实例(可选)
                如果提供,将在数据集层面预处理输入图像,避免训练时的转换开销
             device: torch.device 或字符串(可选)
                如果提供,预处理后的 tensor 将直接移动到该设备(通常是 GPU)
                如果为 None,将自动检测:如果 CUDA 可用则使用当前 CUDA 设备,否则使用 CPU
            num_workers: DataLoader 的 num_workers(可选)
                如果提供,将用于正确计算每个 worker 的 epoch 大小
            batch_size: batch_size(可选)
                如果提供,将用于更精确地计算 epoch 大小
        """
        super().__init__()
        self.task = task
        self.train_image_root = train_image_root
        self.vis_image_root = vis_image_root
        self.train_tar_pattern = train_tar_pattern
        self.test_tar_pattern = test_tar_pattern
        self.estimated_samples_per_shard = estimated_samples_per_shard
        self.split_data_by_node = split_data_by_node
        self.vl_chat_processor = vl_chat_processor
        self.device = device
        self.fid_stat_path = fid_stat_path
        
        print('Creating online dataset (loads images only, no encoding)...')
        
        # 判断使用哪种数据集
        if train_tar_pattern is not None:
            # 使用 WebDataset 模式
            print(f'Using WebDataset mode with pattern: {train_tar_pattern}')
            
            self.train = WebDatasetDataset(
                tar_pattern=train_tar_pattern,
                resolution=resolution,
                shuffle_buffer=shuffle_buffer,
                resampled=resampled,
                split_data_by_node_flag=split_data_by_node,
                estimated_samples_per_shard=estimated_samples_per_shard,
                vl_chat_processor=vl_chat_processor,
                device=device,
                num_workers=num_workers,
                batch_size=batch_size
            )
            self.test = WebDatasetDataset(
                tar_pattern=test_tar_pattern,
                resolution=resolution,
                shuffle_buffer=100, 
                resampled=True, 
                split_data_by_node_flag=split_data_by_node,
                allow_shared_shards=True,  # 允许多个进程共享shard(当shard数量少于进程数时)
                estimated_samples_per_shard=estimated_samples_per_shard,
                vl_chat_processor=vl_chat_processor,
                device=device,
                num_workers=num_workers,
                batch_size=batch_size
            )
        elif train_image_root is not None:
            # 使用文件系统模式(原有逻辑)
            print(f'Using filesystem mode with root: {train_image_root}')
            full_dataset = OnlineDataset(
                image_root=train_image_root,
                resolution=resolution,
                task=task
            )
            
            # 划分训练集和验证集
            total_len = len(full_dataset)
            val_len = int(total_len * val_split_ratio)
            train_len = total_len - val_len
            
            # 使用固定种子确保划分可复现
            generator = torch.Generator().manual_seed(42)
            self.train, self.test = torch.utils.data.random_split(
                full_dataset, [train_len, val_len], generator=generator
            )
        else:
            raise ValueError("必须提供 train_image_root 或 train_tar_pattern 之一")
        
        print(f'Dataset ready: train={len(self.train) if hasattr(self.train, "__len__") else "unknown"}, '
              f'val={len(self.test) if hasattr(self.test, "__len__") else "unknown"}')
        assert not cfg
        self.resolution = resolution

        # 收集vis_image_root/input下的图片路径(用于生成)
        # 同时收集vis_image_root/output下的图片路径(用于ground truth)
        # 注意:这里存储路径列表,而不是PIL Image,因为vis_image_paths可能包含大量图像
        # 在需要时按需加载,避免占用过多内存
        self.vis_image_paths = []  # input 图片路径
        self.vis_output_paths = []  # output 图片路径
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        
        if os.path.exists(self.vis_image_root):
            input_dir = os.path.join(self.vis_image_root, 'input')
            output_dir = os.path.join(self.vis_image_root, 'output')
            
            # 收集 input 目录下的图片路径
            if os.path.exists(input_dir):
                print(f"Scanning input images in: {input_dir}")
                for root, dirs, files in os.walk(input_dir):
                    for filename in files:
                        if any(filename.endswith(ext) for ext in valid_extensions):
                            full_path = os.path.join(root, filename)
                            self.vis_image_paths.append(full_path)
                self.vis_image_paths = sorted(self.vis_image_paths)
                print(f"Found {len(self.vis_image_paths)} input images")
            else:
                print(f"Warning: input directory does not exist: {input_dir}")
            
            # 收集 output 目录下的图片路径
            if os.path.exists(output_dir):
                print(f"Scanning output images in: {output_dir}")
                for root, dirs, files in os.walk(output_dir):
                    for filename in files:
                        if any(filename.endswith(ext) for ext in valid_extensions):
                            full_path = os.path.join(root, filename)
                            self.vis_output_paths.append(full_path)
                self.vis_output_paths = sorted(self.vis_output_paths)
                print(f"Found {len(self.vis_output_paths)} output images")
            else:
                print(f"Warning: output directory does not exist: {output_dir}")
            
            # 确保 input 和 output 图片数量一致(通过文件名匹配)
            if len(self.vis_image_paths) > 0 and len(self.vis_output_paths) > 0:
                # 创建文件名到路径的映射
                input_map = {}
                for path in self.vis_image_paths:
                    basename = os.path.basename(path)
                    # 去除可能的扩展名差异,只保留基础文件名
                    base_name = os.path.splitext(basename)[0]
                    input_map[base_name] = path
                
                output_map = {}
                for path in self.vis_output_paths:
                    basename = os.path.basename(path)
                    base_name = os.path.splitext(basename)[0]
                    output_map[base_name] = path
                
                # 找到匹配的图片对
                matched_keys = sorted(set(input_map.keys()) & set(output_map.keys()))
                self.vis_image_paths = [input_map[key] for key in matched_keys]
                self.vis_output_paths = [output_map[key] for key in matched_keys]
                print(f"Matched {len(self.vis_image_paths)} input-output image pairs")
            
            print(f"Note: vis_image_paths stores paths (not loaded images) to save memory.")
            print(f"Images will be loaded on-demand when needed.")
        else:
            print(f"Warning: vis_image_root does not exist: {self.vis_image_root}")
            self.vis_image_paths = []
            self.vis_output_paths = []
        
        # 可选:提供方法来获取PIL Image列表(用于批量处理)
        # 注意:只在需要时使用,避免占用过多内存
        self._vis_image_pil_cache = None
    
    def get_vis_images_as_pil(self, max_images=None):
        """
        获取vis_image_paths对应的PIL Image列表(使用多线程并行加载)
        
        Args:
            max_images: 最大图像数量(None表示全部)
        
        Returns:
            PIL Image列表
        """
        from PIL import Image
        import concurrent.futures
        
        paths = self.vis_image_paths[:max_images] if max_images else self.vis_image_paths
        
        if not paths:
            return []
        
        def load_image(image_path):
            """加载单个图像"""
            try:
                pil_img = Image.open(image_path)
                pil_img.load()  # 强制加载图像数据
                return pil_img.convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load vis image {image_path}: {e}")
                # 创建占位图像
                return Image.new('RGB', (384, 384), color='black')
        
        # 使用多线程并行加载图像(I/O密集型操作,可以显著加速)
        max_workers = min(len(paths), 8)
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                pil_images = list(executor.map(load_image, paths))
        else:
            # 单线程情况
            pil_images = [load_image(path) for path in paths]
        
        return pil_images
    
    def get_vis_images_as_tensor(self, max_images=None, vl_chat_processor=None):
        """
        获取vis_image_paths对应的预处理后的tensor(使用多线程并行加载)
        
        Args:
            max_images: 最大图像数量(None表示全部)
            vl_chat_processor: VLChatProcessor实例,如果提供则返回预处理后的tensor [batch_size, 3, H, W]
        
        Returns:
            torch.Tensor: 形状为 [batch_size, 3, H, W] 的tensor(如果提供了vl_chat_processor)
            或 PIL Image列表(如果未提供vl_chat_processor)
        """
        from PIL import Image
        import concurrent.futures
        
        paths = self.vis_image_paths[:max_images] if max_images else self.vis_image_paths
        
        if not paths:
            return torch.empty((0, 3, 256, 256)) if vl_chat_processor else []
        
        def load_image(image_path):
            """加载单个图像"""
            try:
                pil_img = Image.open(image_path)
                pil_img.load()  # 强制加载图像数据
                return pil_img.convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load vis image {image_path}: {e}")
                # 创建占位图像
                return Image.new('RGB', (384, 384), color='black')
        
        # 使用多线程并行加载图像(I/O密集型操作,可以显著加速)
        max_workers = min(len(paths), 8)
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                pil_images = list(executor.map(load_image, paths))
        else:
            # 单线程情况
            pil_images = [load_image(path) for path in paths]
        
        # 如果提供了vl_chat_processor,返回预处理后的tensor
        if vl_chat_processor is not None:
            images_outputs = vl_chat_processor.image_processor(pil_images, return_tensors="pt")
            return images_outputs.pixel_values  # [batch_size, 3, H, W]
        else:
            # 否则返回PIL Image列表(向后兼容)
            return pil_images
    
    def get_vis_output_images_as_pil(self, max_images=None):
        """
        获取vis_output_paths对应的PIL Image列表(使用多线程并行加载)
        
        Args:
            max_images: 最大图像数量(None表示全部)
        
        Returns:
            PIL Image列表
        """
        from PIL import Image
        import concurrent.futures
        
        paths = self.vis_output_paths[:max_images] if max_images else self.vis_output_paths
        
        if not paths:
            return []
        
        def load_image(image_path):
            """加载单个图像"""
            try:
                pil_img = Image.open(image_path)
                pil_img.load()  # 强制加载图像数据
                return pil_img.convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load vis output image {image_path}: {e}")
                # 创建占位图像
                return Image.new('RGB', (384, 384), color='black')
        
        # 使用多线程并行加载图像(I/O密集型操作,可以显著加速)
        max_workers = min(len(paths), 8)
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                pil_images = list(executor.map(load_image, paths))
        else:
            # 单线程情况
            pil_images = [load_image(path) for path in paths]
        
        return pil_images

    @property
    def data_shape(self):
        if self.resolution==512:
            return 4, 64, 64
        else:
            return 4, 32, 32
    
    @property
    def fid_stat(self):
        # 如果通过命令行或配置传入了 fid_stat_path,则使用它；否则使用默认路径
        if self.fid_stat_path:
            return self.fid_stat_path
        return f'/storage/v-jinpewang/lab_folder/qisheng_data/assets/fid_stats/fid_stats_mscoco256_val.npz'

def get_dataset(name, **kwargs):
    if name == 'JDB_demo_features':
        return JDBFullFeatures(**kwargs)
    elif name == 'textimage_features':
        return TextImageFeatures(**kwargs)
    elif name == 'online_features':
        return OnlineFeatures(**kwargs)
    else:
        raise NotImplementedError(name)
