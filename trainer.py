#!/usr/local/bin/python3

import sys

sys.path.append('./open_source_model/LLaVA-main')  # Replace it with your llava path
sys.path.append('./open_source_model/InstructBlip')  # Replace it with your instructblip path
sys.path.append('./open_source_model/mPLUG-Owl-main/mPLUG-Owl2')  # Replace it with your mplug-owl2 path

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import log
import json
import copy
import torch
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
from corenlp_client import CoreNLP
from torch.utils.data import DataLoader
from transformers import StoppingCriteria, AutoTokenizer
from llava.model import *
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from functools import partial
import torch.nn.functional as F


def create_dir(name):
    if not os.path.exists(name):
        os.makedirs(name)


def bilinear_interpolation(img, height, width):
    """
	bilinear interpolation
	:param img: original image
	:param height: image height after scaling
	:param width: image width after scaling
	:return: image after scaling
	"""
    # height and width of original image
    ori_height, ori_width = img.shape[:2]
    # image after scaling
    dst = np.zeros((height, width, img.shape[2]))
    # scaling factor
    scale_x = float(ori_width) / width
    scale_y = float(ori_height) / height

    for i in range(height):
        for j in range(width):
            x = min(int(j * scale_x), ori_width - 2)
            y = min(int(i * scale_y), ori_height - 2)
            u = j * scale_x - x
            v = i * scale_y - y
            dst[i, j] = (1 - u) * (1 - v) * img[y, x] + \
                        u * (1 - v) * img[y, x + 1] + \
                        (1 - u) * v * img[y + 1, x] + \
                        u * v * img[y + 1, x + 1]
    return dst


def resize_and_crop_like_clip(image, target_size=224):
    """
    完全复制CLIPImageProcessor的方式：
    1. 调整最短边到target_size，保持长宽比
    2. 中心裁剪到target_size×target_size
    """
    from PIL import Image

    # 1. 调整最短边（保持长宽比）
    w, h = image.size

    # 计算缩放比例
    if w < h:  # 宽度是短边
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:  # 高度是短边
        new_h = target_size
        new_w = int(w * (target_size / h))

    # 使用BICUBIC插值（与处理器一致）
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # 2. 中心裁剪
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    cropped = resized.crop((left, top, right, bottom))
    return cropped


class trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.annotator = CoreNLP(
            url="http://localhost:9000",  # 默认端口
            timeout=30000,  # 超时时间
        )
        self.getModel()
        logger = log.get_logger(args.save_to + "/log")
        self.logger = logger
        logger.info(args)

   # ===================== 钩子：在 o_proj 输入处捕获每个 head 的原始子向量 =====================
    def _capture_head_inputs_llava(self, layer_idx, module, args, kwargs=None):
        """
        hook o_proj 的 forward_pre_hook：args[0] 形如 [B, T, num_heads*head_dim]，
        即所有 head 拼接后、W_O 投影前的张量。
        论文公式(6)(7)需要 a~_{h,-1}^l = a_{h,-1}^l W_O。
        注意：调用方需保证本次前向的最后一个 token 就是目标名词 token，
        这样 [:,-1,:] 才与 Q 的 noun_pos 对齐。
        """
        if not args:
            return
        attn_concat = args[0]
        if attn_concat.dim() != 3:
            return

        attn_module = self.model.model.layers[layer_idx].self_attn
        num_heads = attn_module.num_heads
        head_dim = getattr(attn_module, 'head_dim', None)
        if head_dim is None:
            head_dim = attn_module.hidden_size // num_heads

        B, T, H = attn_concat.shape
        if H != num_heads * head_dim:
            return

        # 取最后一个 token（调用方保证它是目标名词 token）
        last_tok = attn_concat[:, -1, :].view(B, num_heads, head_dim)   # [B, Nh, Dh]

        # W_O = o_proj.weight: [H_out, num_heads*head_dim]
        W_O = attn_module.o_proj.weight.detach()                        # [H, Nh*Dh]
        W_O_heads = W_O.view(W_O.shape[0], num_heads, head_dim)         # [H, Nh, Dh]

        for h in range(num_heads):
            a_h = last_tok[:, h, :]                                     # [B, Dh]
            W_O_h = W_O_heads[:, h, :]                                  # [H, Dh]
            # a~_h = a_h @ W_O_h^T  -> [B, H]
            a_h_proj = torch.einsum('bd,od->bo', a_h, W_O_h)
            key = (layer_idx, h)
            self.head_outputs.setdefault(key, []).append(a_h_proj.detach().cpu())

   # ===================== 注意力头重要性：论文公式(6) 归一化 L2 =====================
    def compute_head_importance(self):
        if not self.head_outputs:
            return {}
        head_l2 = {}
        for (layer_idx, head_idx), outputs_list in self.head_outputs.items():
            if not outputs_list:
                continue
            all_out = torch.cat(outputs_list, dim=0)                  # [B, H]
            head_l2[(layer_idx, head_idx)] = all_out.norm(p=2, dim=-1).mean().item()

        # 按层做归一化：alpha_h = ||a~_h||_2 / sum_r ||a~_r||_2   —— 论文公式(6)
        layer_heads = {}
        for (l, h), v in head_l2.items():
            layer_heads.setdefault(l, []).append((h, v))

        alpha = {}
        for l, heads in layer_heads.items():
            denom = sum(v for _, v in heads) + 1e-8
            for h, v in heads:
                alpha[(l, h)] = v / denom
        return alpha

    # ===================== 论文公式(7)(8)(9) 的严格实现 =====================
    def compute_attention_weighted_scores(self, raw_scores, alpha, head_dict, w_out_col, lam=1.0):
        """
        论文 Section 3.B (修改后的公式版本)：
          (6) alpha_h^l  = ||a~_h||_2 / sum_r ||a~_r||_2            —— 由 compute_head_importance 给出
          (7) beta_{i,h}^l = cos( (W_out^l)_{:,i},  a~_{h,-1}^l )   —— (W_out^l)_{:,i} 是 down_proj 的第 i 列
          (8) gamma_i^l = sum_h alpha_h^l * beta_{i,h}^l
          (9) s_{i,j}^l = Q^l(i,j) * (1 + lam * gamma_i^l)
        注：论文式(9)本身不带 lam；lam 是实际训练时的缩放系数，默认取 1.0 更贴近论文。

        参数:
          raw_scores       : [num_layers, num_neurons]       式(5) 的 Q^l(i,j)
          alpha            : {(layer, head): alpha_h^l}
          head_dict        : {(layer, head): [tensor(B, H)]} —— 由 _capture_head_inputs_llava 填入
          w_out_col        : [num_layers, num_neurons, H]    FFN 的 down_proj 权重的转置（即每行是一个列向量 W_out[:, i]）
        """
        num_layers, num_neurons = raw_scores.shape
        device = raw_scores.device
        eps = 1e-8
        final = raw_scores.clone().float()

        # 按层聚合每个 head 的 (a~_h, alpha_h)
        layer_cache = {}
        for (l, h), weight in alpha.items():
            if l not in layer_cache:
                layer_cache[l] = ([], [])
            try:
                feat = head_dict[(l, h)][-1]                          # 最后一个 token 的 a~_h^l
                feat = feat.squeeze(0).to(device).float()             # [H]
                if feat.numel() == 0 or feat.norm() < eps:
                    continue
                layer_cache[l][0].append(feat)
                layer_cache[l][1].append(weight)
            except (KeyError, IndexError):
                continue

        for l in range(num_layers):
            if l not in layer_cache or l >= w_out_col.shape[0]:
                continue

            head_feats, head_alphas = layer_cache[l]
            if not head_feats:
                continue

            # W_out_col^l: [num_neurons, H] (由 down_proj.T 得到，第 i 行即 W_out[:, i])
            W_out_l = w_out_col[l].to(device).float()
            W_norm = W_out_l.norm(p=2, dim=1).clamp_min(eps)            # [d]

            gamma = torch.zeros(num_neurons, device=device, dtype=torch.float32)

            for feat, alpha_h in zip(head_feats, head_alphas):
                feat = feat.to(device).float()                         # [H]
                feat_norm = feat.norm(p=2).clamp_min(eps)

                # 公式(7): β = cos(W_out 的第 i 列, a~_h)
                dot = torch.matmul(W_out_l, feat)                      # [d]
                beta = dot / (W_norm * feat_norm)                      # [d]
                beta = torch.clamp(beta, -0.999, 0.999)

                # 公式(8): γ += α_h · β_{i,h}
                gamma += alpha_h * beta

            # 公式(9): s = Q · (1 + λ·γ)
            q = raw_scores[l].float()
            final[l] = q * (1.0 + lam * gamma)

        final = torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
        return final.half()

    # ===================== 每个名词单独保存 =====================
    def save_head_importance_for_noun(self, head_alpha, noun_dir):
        if not head_alpha:
            return
        importance_dir = f'{noun_dir}/head_importance'
        create_dir(importance_dir)

        importance_json = {}
        for (l, h), v in head_alpha.items():
            importance_json[f"layer_{l}_head_{h}"] = v

        with open(f'{importance_dir}/head_alpha.json', 'w', encoding='utf-8') as f:
            json.dump(importance_json, f, indent=2)

        if head_alpha:
            max_l = max(l for l, _ in head_alpha.keys()) + 1
            max_h = max(h for _, h in head_alpha.keys()) + 1
            mat = torch.zeros(max_l, max_h)
            for (l, h), v in head_alpha.items():
                mat[l, h] = v
            torch.save(mat, f'{importance_dir}/head_alpha_matrix.pt')

    def getModel(self):
        # Getting the required model
        if self.args.model_type == "llava":
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
                DEFAULT_IM_END_TOKEN

            # load llava
            self.model_name = get_model_name_from_path(self.args.model_path)
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.args.model_path,
                                                                                                  self.args.model_base,
                                                                                                  self.model_name)
            print('Successfully load.')

            # get query
            qs = self.args.query
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if 'llama-2' in self.model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in self.model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                    conv_mode, self.args.conv_mode, self.args.conv_mode))
            else:
                self.args.conv_mode = conv_mode

            self.conv = conv_templates[self.args.conv_mode].copy()
            self.conv.append_message(self.conv.roles[0], qs)
            self.conv.append_message(self.conv.roles[1], None)
            prompt = self.conv.get_prompt()

            # add an instruction string for better generation
            if self.args.add_prompt:
                prompt += 'An image of'

            # get input ids
            self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                   return_tensors='pt').unsqueeze(0).cuda()

            # image token length of llava is 256/441
            self.image_token_len = 256

            patch_cut = prompt[:prompt.find(DEFAULT_IMAGE_TOKEN)]
            self.patch_start = len(self.tokenizer.encode(patch_cut))
            self.patch_end = self.patch_start + self.image_token_len

        elif self.args.model_type == "instructblip":
            from modeling_instructblip import InstructBlipForConditionalGeneration
            from processing_instructblip import InstructBlipProcessor

            self.model = InstructBlipForConditionalGeneration.from_pretrained(self.args.model_path,
                                                                              low_cpu_mem_usage=True, device_map="auto")
            self.processor = InstructBlipProcessor.from_pretrained(self.args.model_path, low_cpu_mem_usage=True,
                                                                   device_map="auto")
            self.tokenizer = self.processor.tokenizer

            self.prompt = self.args.query
            if self.args.add_prompt:
                self.prompt += ' An image of'

            self.image_token_len = 32

        elif self.args.model_type == "mplug-owl2":
            from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            from mplug_owl2.conversation import conv_templates, SeparatorStyle
            from mplug_owl2.model.builder import load_pretrained_model
            from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
                KeywordsStoppingCriteria

            self.model_name = get_model_name_from_path(self.args.model_path)
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.args.model_path,
                                                                                                  self.args.model_base,
                                                                                                  self.model_name)
            print('Successfully load.')

            inp = DEFAULT_IMAGE_TOKEN + self.args.query

            self.conv = conv_templates["mplug_owl2"].copy()
            self.conv.append_message(self.conv.roles[0], inp)
            self.conv.append_message(self.conv.roles[1], None)
            prompt = self.conv.get_prompt()

            # add an instruction string for better generation
            if self.args.add_prompt:
                prompt += ' An image of'

            print(prompt)

            # get input ids
            self.input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                                   return_tensors='pt').unsqueeze(0).cuda()

            self.image_token_len = 65

    def process_batch(self, batch):
        if "sbu" in self.args.task_type:
            image = Image.fromarray(np.uint8(batch['image'][0].numpy().transpose((1, 2, 0)) * 255))
            label = None
        else:
            raise NotImplementedError

        return image, label

    def perturbation(self, eval_data):
        test_dataloader = DataLoader(eval_data, batch_size=self.args.bz, num_workers=1, pin_memory=True)
        cnt = 0
        self.model.eval()

        for data in tqdm(test_dataloader):
            cnt += 1
            if cnt < self.args.start:
                continue

            print(f'cnt: {cnt}')

            noun_paths = glob(f'{self.args.save_to}/{cnt}/noun_*')

            with open(f'{noun_paths[0]}/data.json') as f:
                original_outputs = json.load(f)['caption']

            image, _ = self.process_batch(data)

            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            # randomly select neurons
            _random = torch.randperm(self.model.config.intermediate_size * self.model.config.num_hidden_layers)[
                      :self.args.top_neurons]
            random_mm_neurons = [[x // self.model.config.intermediate_size, x % self.model.config.intermediate_size] for
                                 x in _random]

            # modify model state dict
            state_dict = self.model.state_dict()
            original_w = []

            for i, mm_neuron in enumerate(random_mm_neurons):
                w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                original_w.append(copy.deepcopy(w[mm_neuron[1]]))
                w[mm_neuron[1]] += torch.randn(w.shape[1]).to(w.device) * self.args.sigma
                state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

            self.model.load_state_dict(state_dict)

            with torch.no_grad():
                output_ids = self.model.generate(
                    self.input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    max_new_tokens=50,
                    stopping_criteria=[stopping_criteria])

            input_token_len = self.input_ids.shape[1]
            print(output_ids[:, input_token_len:])
            n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            random_outputs = outputs.strip()

            print(f'random outputs: {random_outputs}')

            # reset model state dict
            for i, mm_neuron in enumerate(random_mm_neurons):
                w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                w[mm_neuron[1]] = original_w[i]
                state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

            self.model.load_state_dict(state_dict)

            for noun_path in noun_paths:
                print(f'noun: {noun_path.split("_")[-1]}')

                mm_neurons = torch.load(f'{noun_path}/mm_neurons')[:self.args.top_neurons]

                # modify model state dict
                state_dict = self.model.state_dict()
                original_w = []

                for i, mm_neuron in enumerate(mm_neurons):
                    w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                    original_w.append(copy.deepcopy(w[mm_neuron[1]]))
                    w[mm_neuron[1]] += torch.randn(w.shape[1]).to(w.device) * self.args.sigma
                    state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

                self.model.load_state_dict(state_dict)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        self.input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=50,
                        stopping_criteria=[stopping_criteria])

                input_token_len = self.input_ids.shape[1]
                print(output_ids[:, input_token_len:])
                n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                print(f'original outputs: {original_outputs}')
                print(f'perturb outputs: {outputs}')

                results = {'original outputs': original_outputs, 'perturb outputs': outputs,
                           'random outputs': random_outputs}

                with open(f'{noun_path}/perturb.json', 'w+', encoding='utf-8') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))

                # reset model state dict
                for i, mm_neuron in enumerate(mm_neurons):
                    w = state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'].T
                    w[mm_neuron[1]] = original_w[i]
                    state_dict[f'model.layers.{mm_neuron[0]}.mlp.down_proj.weight'] = w.T

                self.model.load_state_dict(state_dict)

            if 0 < self.args.max_num <= cnt - self.args.start + 1 or self.args.end <= cnt:
                break

    def find_mm_neurons(self, eval_data):
        test_dataloader = DataLoader(eval_data, batch_size=self.args.bz, num_workers=1, pin_memory=True)
        cnt = 0
        self.model.eval()

        if self.args.model_type == 'llava':
            total_top_words = torch.load(f'{self.args.preparation_path}/llava_13b_neuron_tokens')
            projection = [torch.load(f'{self.args.preparation_path}/llava_13b_layer_projection/layer_{layer}').cpu() for
                          layer in tqdm(range(self.model.config.num_hidden_layers))]
        elif self.args.model_type == 'instructblip':
            total_top_words = torch.load(f'{self.args.preparation_path}/instructblip_7b_neuron_tokens')
            projection = [
                torch.load(f'{self.args.preparation_path}/instructblip_7b_layer_projection/layer_{layer}').cpu() for
                layer in
                tqdm(range(self.model.config.num_hidden_layers))]
        elif self.args.model_type == 'mplug-owl2':
            total_top_words = torch.load(f'{self.args.preparation_path}/mplug_owl2_7b_neuron_tokens')
            projection = [torch.load(f'{self.args.preparation_path}/mplug_owl2_7b_layer_projection/layer_{layer}').cpu()
                          for layer in
                          tqdm(range(self.model.config.num_hidden_layers))]

        # [NEW] 构造 W_out 列的集合：[L, d, H]，对应论文式(7)的 W_out^l 的第 i 列
        # down_proj.weight 形状为 [H, d]，转置后形状为 [d, H]，其第 i 行正好是原矩阵的第 i 列
        if self.args.model_type == 'llava':
            w_out_col = torch.stack([
                self.model.model.layers[l].mlp.down_proj.weight.detach().float().cpu().T
                for l in range(self.model.config.num_hidden_layers)
            ])
            print(f'[NEW] W_out (column representation) stack shape: {w_out_col.shape}')
        else:
            w_out_col = None

        import time
        start = time.perf_counter()

        for data in tqdm(test_dataloader):
            cnt += 1
            if cnt < self.args.start:
                continue

            create_dir(f'{self.args.save_to}/{cnt}')

            if self.args.cal_activations:
                self.model.set_get_activations(self.patch_start, self.patch_end)

            image, _ = self.process_batch(data)

            image.save(f'{self.args.save_to}/{cnt}/original_image.png')

            if self.args.model_type == 'llava':
                # 1. 获取用于模型输入的图像张量（标准化后的）
                result = self.image_processor.preprocess(image, return_tensors='pt')
                image_tensor = result['pixel_values'][0]  # (3, 224, 224) 用于模型

                # 2. 关键：获取用于热力图的原始图像张量
                print(f"图像处理器返回的键: {list(result.keys())}")

                if 'pixel' in result:
                    img_tensor = result['pixel'][0]
                    print(f"使用 'pixel' 键获取图像张量: {img_tensor.shape}")
                elif 'original_pixel_values' in result:
                    img_tensor = result['original_pixel_values'][0]
                    print(f"使用 'original_pixel_values' 键获取图像张量: {img_tensor.shape}")
                else:
                    print("警告：未找到原始图像张量，使用原始图像创建")
                    target_size = (224, 224)
                    image = resize_and_crop_like_clip(image, target_size[0])
                    img_array = np.array(image).astype(np.float32)
                    img_tensor = torch.from_numpy(img_array)  # (H, W, C)
                    print(f"从原始图像创建张量: {img_tensor.shape}")

                if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.squeeze(0)

                print(f"最终图像张量形状: {img_tensor.shape}, 值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                print(image_tensor.shape)

                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            elif self.args.model_type == 'instructblip':
                image_tensor = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                img_tensor = self.processor.image_processor.preprocess(image, return_tensors='pt')['pixel'][0]

                stop_str = ""

            elif self.args.model_type == 'mplug-owl2':
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                img_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel'][0]

                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            if self.args.shuffle:
                self.model.set_shuffle()
                idx = torch.randperm(self.image_token_len)
                print(idx)
                torch.save(idx.cpu(), f'{self.args.save_to}/{cnt}/idx')
                idx_dict = {i: x.item() for i, x in enumerate(idx)}
                idx_restore_dict = {x.item(): i for i, x in enumerate(idx)}
                idx_restore = torch.tensor([idx_restore_dict[x] for x in range(self.image_token_len)])
                self.model.set_idx(idx)

            with torch.no_grad():
                if self.args.model_type == 'llava':
                    output_ids = self.model.generate(
                        self.input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=50,
                        stopping_criteria=[stopping_criteria])

                elif self.args.model_type == "instructblip":
                    self.inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
                    for k, v in self.inputs.items():
                        self.inputs[k] = v.cuda()

                    self.input_ids = self.inputs['input_ids']

                    output_ids = self.model.generate(
                        **self.inputs,
                        max_new_tokens=50,
                    )

                elif self.args.model_type == 'mplug-owl2':
                    output_ids = self.model.generate(
                        self.input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        max_new_tokens=50,
                        stopping_criteria=[stopping_criteria])

                input_token_len = self.input_ids.shape[1]
                print(output_ids[:, input_token_len:])
                n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

                output_tokens = output_ids[:, input_token_len:]

                outputs = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if self.args.model_type == 'llava' or self.args.model_type == 'mplug-owl2':
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                print("outputs", outputs)
                output_file = f'{self.args.save_to}/{cnt}/outputs.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(outputs)
                print(f"Outputs saved to: {output_file}")

                output_tokens = output_tokens[0].cpu().tolist()
                print(output_tokens)

            if self.args.cal_activations:
                last_activation = self.model.get_activations()
                last_activation = torch.stack([torch.stack(x) for x in last_activation])
                print(last_activation.shape)

            tokenized_text = self.annotator.tokenize(outputs.replace('.', ' '))[0]
            pos_tag = self.annotator.pos_tag(outputs.replace('.', ' '))[0]
            nouns = list(set(word for word, pos in zip(tokenized_text, pos_tag) if pos.startswith('N')))
            nouns.sort()

            print(nouns)

            noun_tokens = [x[0] for x in self.tokenizer.batch_encode_plus(nouns, add_special_tokens=False)['input_ids']]
            noun_pos = []
            new_nouns = []

            for i, x in enumerate(noun_tokens):
                if x in output_tokens:
                    noun_pos.append(output_tokens.index(x))
                    new_nouns.append(nouns[i])

            nouns = new_nouns

            with open(f'{self.args.save_to}/{cnt}/nouns.json', 'w+', encoding='utf-8') as f:
                f.write(json.dumps(nouns, ensure_ascii=False))

            torch.save(last_activation[noun_pos].cpu(), f'{self.args.save_to}/{cnt}/last_activation')

            for j, (token, noun, pos) in enumerate(zip(noun_tokens, nouns, noun_pos)):
                save_dir = f'{self.args.save_to}/{cnt}/noun_{noun}'
                create_dir(save_dir)

                # ===================== 为当前名词注册 o_proj 的 pre-hook =====================
                self.head_outputs = {}
                hooks = []
                for layer_idx, layer in enumerate(self.model.model.layers):
                    attn = getattr(layer, 'self_attn', None)
                    if attn is not None and hasattr(attn, 'o_proj'):
                        h = attn.o_proj.register_forward_pre_hook(
                            partial(self._capture_head_inputs_llava, layer_idx)
                        )
                        hooks.append(h)

                # ===== 关键修改：第二次前向喂 “prompt + 生成到名词 token 的前缀” =====
                # 使本次前向的最后一个 token 就是目标名词 token，
                # hook 里的 [:,-1,:] 与 Q 的 noun_pos 对齐，满足论文式(6)-(9)同位置假设。
                gen_prefix = output_ids[:, input_token_len:input_token_len + pos + 1]
                prefix_ids = torch.cat([self.input_ids, gen_prefix], dim=1)

                with torch.no_grad():
                    _ = self.model(
                        prefix_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        return_dict=True
                    )

                # 计算并保存该名词专属注意力头重要性（论文公式6：归一化 L2）
                head_alpha = self.compute_head_importance()
                self.save_head_importance_for_noun(head_alpha, save_dir)

                # 清除钩子
                for h in hooks:
                    h.remove()
                torch.cuda.empty_cache()

                # calculate scores of neurons (式(5) 的 Q^l(i,j) 基础贡献)
                scores = torch.zeros((last_activation.shape[1], last_activation.shape[2])).half().cuda()
                for layer in tqdm(range(last_activation.shape[1])):
                    act = last_activation[pos, layer]
                    prob = act.cuda() * projection[layer][:, token].flatten().cuda()
                    scores[layer] += prob.flatten()

                print(noun)

                # ===================== 论文式(7)(8)(9)：注意力加权贡献 =====================
                if w_out_col is not None:
                    scores = self.compute_attention_weighted_scores(
                        scores, head_alpha, self.head_outputs, w_out_col, lam=1.0
                    )
                else:
                    print('[WARN] w_out_col 未构造，跳过注意力加权（仅 LLaVA 支持）')

                top_index = scores.flatten().sort(descending=True, dim=0).indices[
                            :10000]  # save top-10000 multi-modal neurons
                top_pos = [(x.item() // scores.shape[1], x.item() % scores.shape[1]) for x in top_index]

                results = {'caption': outputs, 'top_neurons': []}

                # find multi-modal neurons
                mm_neurons = []
                for i, pos in enumerate(top_pos):
                    mm_neurons.append([pos[0], pos[1]])
                    if i < 10:
                        print(
                            f'Neuron: L{pos[0]}.u{pos[1]} \tScore: {round(scores[pos[0]][pos[1]].item(), 4)}\tTop 10 tokens: {total_top_words[pos[0]][pos[1]][:10]}')
                        results['top_neurons'].append({'neuron': pos, 'tokens': total_top_words[pos[0]][pos[1]][:10],
                                                       'score': scores[pos[0]][pos[1]].item()})

                with open(f'{save_dir}/data.json', 'w+', encoding='utf-8') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))

                torch.save(mm_neurons, f'{save_dir}/mm_neurons')

                # 保存前100/500神经元位置
                self.save_neuron_positions(
                    mm_neurons,
                    scores.cpu(),
                    save_dir,
                    noun_name=noun
                )

                self.save_neuron_contribution_scores(
                    scores=scores,
                    mm_neurons=mm_neurons,
                    save_dir=save_dir,
                    noun_name=noun,
                    top_k_list=[100, 200, 300, 400, 500, 1000, 800]
                )

                torch.save(img_tensor, f'{save_dir}/image_tensor')
                image = Image.fromarray(np.uint8(img_tensor.numpy())).convert('RGBA')
                image.save(f'{save_dir}/image.png')

                if self.args.model_type == 'llava':
                    activations = torch.stack(self.model.activations)[:, self.patch_start:self.patch_end]

                    # draw heatmap and binary mask
                    for mean_num in [1, 10, 50, 100, 500, 1000]:
                        mean_activation = torch.stack(
                            [activations[x[0], :, x[1]].reshape(16, 16, 1) for x in mm_neurons[:mean_num]]).mean(
                            dim=0).numpy()

                        new_activation = bilinear_interpolation(mean_activation, img_tensor.shape[0],
                                                                img_tensor.shape[1])

                        new_activation_ = (new_activation - new_activation.min()) / (
                                new_activation.max() - new_activation.min())
                        heatmap = cv2.applyColorMap(np.uint8(255 * new_activation_), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        heatmap = Image.fromarray(heatmap).convert('RGBA')
                        heatmap.putalpha(int(0.5 * 255))
                        new_image = Image.alpha_composite(image, heatmap).convert('RGB')
                        new_image.save(f'{save_dir}/heatmap_average_{mean_num}.png')

                        thres = np.percentile(new_activation, 95)
                        pos = np.where(new_activation > thres)
                        mask = Image.new('RGBA', (img_tensor.shape[0], img_tensor.shape[1]), (0, 0, 0, 200))
                        mask_pixels = mask.load()
                        for i, j in zip(pos[0], pos[1]):
                            mask_pixels[j, i] = (255, 255, 255, 0)
                        new_image = Image.alpha_composite(image, mask).convert('RGB')
                        new_image.save(f'{save_dir}/mask_average_{mean_num}.png')

                    for i in range(5):
                        mm_neuron = mm_neurons[i]
                        torch.save(activations[mm_neuron[0], :, mm_neuron[1]].float().half(),
                                   f'{save_dir}/activation_L{mm_neuron[0]}_u{mm_neuron[1]}')
                        activation = activations[mm_neuron[0], :, mm_neuron[1]].reshape(16, 16, 1).numpy()
                        new_activation = bilinear_interpolation(activation, img_tensor.shape[0], img_tensor.shape[1])

                        new_activation_ = (new_activation - new_activation.min()) / (
                                new_activation.max() - new_activation.min())
                        heatmap = cv2.applyColorMap(np.uint8(255 * new_activation_), cv2.COLORMAP_JET)
                        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        heatmap = Image.fromarray(heatmap).convert('RGBA')
                        heatmap.putalpha(int(0.5 * 255))
                        new_image = Image.alpha_composite(image, heatmap).convert('RGB')
                        new_image.save(f'{save_dir}/heatmap_L{mm_neuron[0]}_u{mm_neuron[1]}.png')

                        thres = np.percentile(new_activation, 95)
                        pos = np.where(new_activation > thres)
                        mask = Image.new('RGBA', (img_tensor.shape[0], img_tensor.shape[1]), (0, 0, 0, 200))
                        mask_pixels = mask.load()
                        for i, j in zip(pos[0], pos[1]):
                            mask_pixels[j, i] = (255, 255, 255, 0)
                        new_image = Image.alpha_composite(image, mask).convert('RGB')
                        new_image.save(f'{save_dir}/mask_L{mm_neuron[0]}_u{mm_neuron[1]}.png')

            if 0 < self.args.max_num <= cnt - self.args.start + 1 or self.args.end <= cnt:
                break

        end = time.perf_counter()
        print(f"avg time: {(end - start) / cnt}")

    def save_neuron_positions(self, mm_neurons, scores, save_dir, noun_name=""):
        """
        保存前100和前500个神经元的位置到txt文件
        mm_neurons: 神经元列表，格式为[[layer1, neuron1], [layer2, neuron2], ...]
        scores: 得分矩阵，shape为[num_layers, num_neurons]
        save_dir: 保存目录
        noun_name: 可选，名词名称，用于文件名
        """
        counts_to_save = [100, 500, 200, 300, 400, 1000, 800]

        for count in counts_to_save:
            current_neurons = mm_neurons[:min(count, len(mm_neurons))]

            if noun_name:
                txt_file = f'{save_dir}/neurons_top{count}_{noun_name}.txt'
            else:
                txt_file = f'{save_dir}/neurons_top{count}.txt'

            print(f"\n保存前 {count} 个神经元位置到: {txt_file}")

            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"# 前 {count} 个多模态神经元位置\n")
                if noun_name:
                    f.write(f"# 名词: {noun_name}\n")
                f.write(f"# 格式: 排名 层号 神经元编号 得分\n")
                f.write("=" * 50 + "\n")

                for i, (layer_idx, neuron_idx) in enumerate(current_neurons):
                    neuron_score = scores[layer_idx][neuron_idx].item()
                    f.write(f"{layer_idx}_{neuron_idx}\n")

        print(f"神经元位置已保存到 {save_dir} 目录")
        return counts_to_save

    def save_neuron_contribution_scores(self, scores, mm_neurons, save_dir, noun_name="", top_k_list=None):
        """
        仅保存TopK神经元的贡献值分数到TXT文件（按分数从高到低）
        新增：每行添加归一化分数（线性归一化，最大值=1）
        """
        score_dir = f"{save_dir}/contribution_scores"
        create_dir(score_dir)

        neuron_scores = []
        for layer_idx, neuron_idx in mm_neurons:
            score_value = scores[layer_idx][neuron_idx].item()
            neuron_scores.append({
                "position": f"{layer_idx}_{neuron_idx}",
                "layer": layer_idx,
                "neuron": neuron_idx,
                "score": score_value
            })

        for top_k in top_k_list:
            current_top_k = min(top_k, len(neuron_scores))
            top_neurons = neuron_scores[:current_top_k]

            if current_top_k > 0:
                max_score = max([item["score"] for item in top_neurons])
                max_score = max_score if max_score > 0 else 1e-8
            else:
                max_score = 1e-8

            txt_file = f"{score_dir}/top{top_k}_scores_{noun_name}.txt"

            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"# Top{current_top_k} 神经元贡献值分数（{noun_name}）\n")
                f.write(f"# 格式：排名  神经元位置  原始分数（保留6位小数）  归一化分数（保留6位小数）\n")
                f.write("#" * 80 + "\n")
                for idx, item in enumerate(top_neurons):
                    normalized_score = item["score"] / max_score
                    f.write(f"{item['position']}\t{normalized_score:.6f}\n")

            print(f"✅ Top{current_top_k} 分数已保存至: {txt_file}")

        return score_dir

    def _algorithm1_knowledge_editing_with_real_activations(self, state_dict, mm_neurons, W_u,
                                                             source_token_id, target_token_id,
                                                             source_noun_path, image_tensor, stop_str, stopping_criteria):
        """
        使用真实激活值的论文算法1实现
        """
        beta = 4.0

        v_0 = W_u[source_token_id].clone()
        v_1 = W_u[target_token_id].clone()

        sample_dir = os.path.dirname(source_noun_path)
        last_activation_path = os.path.join(sample_dir, 'last_activation')
        if os.path.exists(last_activation_path):
            try:
                last_activation = torch.load(last_activation_path)
                print(f"成功加载全局激活数据: {last_activation.shape}")
            except Exception as e:
                last_activation = None
                print(f"加载全局激活数据失败: {e}")
        else:
            last_activation = None
            print("没找到last_activation")

        state_dict = self.model.state_dict()
        original_w = []
        for neuron_idx, (layer_idx, neuron_pos) in enumerate(mm_neurons):
            print(f"\n处理神经元 {neuron_idx + 1}/{len(mm_neurons)}: L{layer_idx}.U{neuron_pos}")

            w_key = f'model.layers.{layer_idx}.mlp.down_proj.weight'
            w_tensor = state_dict[w_key]

            print(f"  权重矩阵原始形状: {w_tensor.shape}")
            print(f"  神经元位置: {neuron_pos}")
            if neuron_pos >= w_tensor.shape[1]:
                print(f"  错误: 神经元位置 {neuron_pos} 超出范围 (最大: {w_tensor.shape[1] - 1})")
                continue
            w = w_tensor.T.clone()
            print(f"  转置后形状: {w.shape}")

            o_i_l = 1.0

            if last_activation is not None:
                try:
                    noun_name = os.path.basename(source_noun_path).split('_')[1]

                    base_dir = os.path.dirname(source_noun_path)
                    cnt_dir = os.path.dirname(base_dir) if 'noun_' in base_dir else base_dir
                    nouns_json_path = os.path.join(cnt_dir, 'nouns.json')

                    if os.path.exists(nouns_json_path):
                        with open(nouns_json_path, 'r', encoding='utf-8') as f:
                            nouns_list = json.load(f)

                        if noun_name in nouns_list:
                            noun_index = nouns_list.index(noun_name)

                            if (noun_index < last_activation.shape[0] and
                                    layer_idx < last_activation.shape[1] and
                                    neuron_pos < last_activation.shape[2]):
                                o_i_l = last_activation[noun_index, layer_idx, neuron_pos].item()
                                print(f"  从last_activation加载激活值: {o_i_l:.6f} (名词: {noun_name}, 索引: {noun_index})")
                            else:
                                print(f"  索引超出范围: noun={noun_index}, layer={layer_idx}, neuron={neuron_pos}")
                                print(f"  last_activation形状: {last_activation.shape}")
                        else:
                            print(f"  名词 '{noun_name}' 不在nouns.json中")
                    else:
                        print(f"  未找到nouns.json: {nouns_json_path}")

                except Exception as e:
                    print(f"  从last_activation提取失败: {e}")

            delta_w = (o_i_l / (2 * beta)) * (v_1 - v_0)

            w_current = w[neuron_pos]
            original_w.append(copy.deepcopy(w_current))
            w_updated = w_current + delta_w.to(w_current.device)
            w[neuron_pos] = w_updated
            state_dict[w_key] = w.T

            original_dot_v0 = torch.dot(w_current, v_0).item()
            original_dot_v1 = torch.dot(w_current, v_1).item()
            updated_dot_v0 = torch.dot(w_updated, v_0).item()
            updated_dot_v1 = torch.dot(w_updated, v_1).item()

            print(f"  激活值 o_i^l = {o_i_l:.6f}")
            print(f"  原始: w·v_0={original_dot_v0:.6f}, w·v_1={original_dot_v1:.6f}")
            print(f"  更新后: w·v_0={updated_dot_v0:.6f}, w·v_1={updated_dot_v1:.6f}")
            print(f"  改进: (w·v_1 - w·v_0)从 {(original_dot_v1 - original_dot_v0):.6f} 变为 {(updated_dot_v1 - updated_dot_v0):.6f}")
            print(f"  Δw范数: {delta_w.norm().item():.6f}")

        self.model.load_state_dict(state_dict)
        with torch.no_grad():
            output_ids = self.model.generate(
                self.input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=50,
                stopping_criteria=[stopping_criteria])

        input_token_len = self.input_ids.shape[1]
        print(output_ids[:, input_token_len:])
        n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        edited_output = outputs.strip()

        print(f'edit_output: {edited_output}')

        for neuron_idx, (layer_idx, neuron_pos) in enumerate(mm_neurons):
            print(f"\n恢复神经元 {neuron_idx + 1}/{len(mm_neurons)}: L{layer_idx}.U{neuron_pos}")
            w_key = f'model.layers.{layer_idx}.mlp.down_proj.weight'
            w_tensor = state_dict[w_key]
            w = w_tensor.T.clone()
            if neuron_pos >= w.shape[0]:
                print(f"  警告: 神经元位置 {neuron_pos} 超出范围，跳过恢复")
                continue

            if neuron_idx < len(original_w):
                w[neuron_pos] = original_w[neuron_idx]
                state_dict[w_key] = w.T
            else:
                print(f"  警告: 没有找到神经元 {neuron_idx} 的原始权重")

        self.model.load_state_dict(state_dict)

        with torch.no_grad():
            output_ids = self.model.generate(
                self.input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                max_new_tokens=50,
                stopping_criteria=[stopping_criteria])

        input_token_len = self.input_ids.shape[1]
        print(output_ids[:, input_token_len:])
        n_diff_input_output = (self.input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        original_output = outputs.strip()

        print(f'orignal outputs: {original_output}')
        return original_output, edited_output

    def knowledge_editing_with_real_activations(self, eval_data, source_concept, target_concept):
        """
        使用真实激活值的知识编辑
        """
        test_dataloader = DataLoader(eval_data, batch_size=self.args.bz, num_workers=1, pin_memory=True)
        cnt = 0

        source_token_id = self.tokenizer.encode(source_concept, add_special_tokens=False)
        target_token_id = self.tokenizer.encode(target_concept, add_special_tokens=False)

        if not source_token_id or not target_token_id:
            print(f"无法编码概念")
            return

        source_token_id = source_token_id[0]
        target_token_id = target_token_id[0]

        print(f"使用真实激活值的知识编辑")
        print(f"源概念: {source_concept} (token_id: {source_token_id})")
        print(f"目标概念: {target_concept} (token_id: {target_token_id})")

        for data in tqdm(test_dataloader):
            cnt += 1
            if cnt < self.args.start:
                continue

            print(f"\n处理样本 {cnt}")

            noun_paths = glob(f'{self.args.save_to}/{cnt}/noun_*')
            source_path = None
            for path in noun_paths:
                if path.split("_")[-1] == source_concept:
                    source_path = path
                    break

            if not source_path:
                print(f"  未找到概念 {source_concept} 的数据")
                continue

            print(f"  找到路径: {source_path}")

            activation_files = glob(f'{source_path}/activation_L*_u*')
            print(f"  找到 {len(activation_files)} 个激活文件")

            try:
                mm_neurons = torch.load(f'{source_path}/mm_neurons')
                print(f"  加载 {len(mm_neurons)} 个神经元")
            except Exception as e:
                print(f"  加载神经元失败: {e}")
                continue

            top_k = min(self.args.top_neurons, len(mm_neurons))
            mm_neurons = mm_neurons[:top_k]

            image, _ = self.process_batch(data)
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, self.input_ids)

            state_dict = self.model.state_dict()

            W_u = None
            for key in ['lm_head.weight', 'language_model.lm_head.weight']:
                if key in state_dict:
                    W_u = state_dict[key]
                    print(f"\嵌入矩阵形状 {W_u.shape}")
                    break

            if W_u is None:
                print("  错误: 未找到嵌入矩阵")
                continue

            original_output, edited_output = self._algorithm1_knowledge_editing_with_real_activations(
                state_dict, mm_neurons, W_u,
                source_token_id, target_token_id,
                source_path, image_tensor, stop_str, stopping_criteria
            )

            results = {
                'sample_id': cnt,
                'source_concept': source_concept,
                'target_concept': target_concept,
                'original_output': original_output,
                'edited_output': edited_output,
                'neurons_used': len(mm_neurons),
                'activation_files_found': len(activation_files)
            }

            result_dir = f'{self.args.save_to}/{cnt}/real_activation_editing'
            create_dir(result_dir)

            result_file = f'{result_dir}/{source_concept}_to_{target_concept}.json'
            with open(result_file, 'w+', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"  原始输出: {original_output}")
            print(f"  编辑后输出: {edited_output}")
            print(f"  结果保存到: {result_file}")

            if 0 < self.args.max_num <= cnt - self.args.start + 1 or self.args.end <= cnt:
                break
