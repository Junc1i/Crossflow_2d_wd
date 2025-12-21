# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# import sys
# import os
# janus_dir = os.path.dirname(os.path.abspath(__file__))
# janus_dir = os.path.abspath(janus_dir)  # 确保绝对路径
# sys.path = [p for p in sys.path if p != janus_dir]  # 过滤掉污染路径
# print(sys.path)

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda("cuda:1").eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>\nConvert the formula into latex code.",
        "images": ["images/equation.png"],
    },  
    {"role": "<|Assistant|>", "content": ""}
]

#generate不支持bz

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device)

# # run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)


# print("inputs_embeds shape:",inputs_embeds.shape)

# # run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
    return_dict_in_generate=True,
    output_hidden_states=True
)

print("len(outputs.hidden_states):",len(outputs.hidden_states))
layers_hidden_states = outputs.hidden_states #512*31*torch.Size([1, 1, 4096]) tuple(tuple(torch.FloatTensor)) to 1*512*4096 last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) — Sequence of hidden-states at the output of the last layer of the model.
#1B 25*2048 7B 31*4096
print("layers_hidden_states[0][-1].shape:",layers_hidden_states[0][-1].shape)

# 1. 计算需要填充的样本数
original_num_samples = len(layers_hidden_states)  
target_num_samples = 512
num_padding_samples = target_num_samples - original_num_samples  

# 2. 创建填充用的零样本
padding_sample = tuple(
    torch.zeros(1, 1, 4096).to(vl_gpt.device)  # 形状 [1, 1, 4096]
    for _ in range(31)
)

# 生成xx个填充样本
padded_samples = [padding_sample for _ in range(num_padding_samples)]


# 3. 合并原始数据与填充数据
combined_samples = list(layers_hidden_states) + padded_samples
# print("combined_samples:",combined_samples)
combined_samples_tuple = tuple(combined_samples)

print("layers_hidden_states[0][-1].shape:",layers_hidden_states[0][-1].shape)
print("list(layers_hidden_states)[0][-1].shape:",list(layers_hidden_states)[0][-1].shape)
print("padded_samples[0][-1].shape:",padded_samples[0][-1].shape)
print("combined_samples[0].shape:",combined_samples[0][-1].shape)
# print(f"合并后的外层维度: {len(combined_samples_tuple)}")          # 应为 512
# print(f"内层维度: {len(combined_samples_tuple[0])}")               # 应为 31
# print(f"张量形状: {combined_samples_tuple[0][0].shape}")           # 应为 torch.Size([1, 1, 4096])

# 4. 生成 attention_mask
attention_mask = [1] * original_num_samples + [0] * num_padding_samples

print("len(combined_samples_tuple):",len(combined_samples_tuple))

print("len(combined_samples_tuple[-1]):", len(combined_samples_tuple[-1]))

print("shape of outputs.sequences[0]:",outputs.sequences[0].shape)

# 1. 提取每个样本的最后一层隐藏状态
last_layer_list = [
    sample_hidden_states[-1]  # 提取第31层的张量（索引-1或30）
    for sample_hidden_states in combined_samples_tuple
]
print("len(last_layer_list):",len(last_layer_list))
# 此时 last_layer_list 是一个包含512个张量的列表，每个张量形状为 [1, 1, 4096]

for i, tensor in enumerate(last_layer_list):
    print(f"Tensor {i} shape: {tensor.shape}")
# 2. 将列表中的张量堆叠为单一张量
# 沿第1维度堆叠（假设需要保持第一个维度为1）


last_layer_list[0] = torch.mean(last_layer_list[0], dim=1, keepdim=True)
stacked_tensor = torch.stack(last_layer_list, dim=1)  # 形状变为 [1, 512, 1, 4096]
print("stacked_tensor.shape:",stacked_tensor.shape)

# 3. 去除多余的维度（第2维的1）
squeezed_tensor = stacked_tensor.squeeze(dim=2)  # 形状变为 [1, 512, 4096]
squeezed_tensor = squeezed_tensor.squeeze(dim=0)  # 形状变为 [512, 4096]
print("squeezed_tensor.shape:",squeezed_tensor.shape)
# 最终结果
final_tensor = squeezed_tensor  # torch.Size([1, 512, 4096])
# 验证形状
print("final_tensor.shape:",final_tensor.shape)  # 输出: torch.Size([1, 512, 4096])
final_tensor= final_tensor.detach().cpu().numpy()

answer = tokenizer.decode(outputs.sequences[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
