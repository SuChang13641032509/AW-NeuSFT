# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    # 新增：按灾害类型配置神经元文件
    neuron_fire_file: Optional[str] = field(default=None)
    neuron_flood_file: Optional[str] = field(default=None)
    neuron_landslide_file: Optional[str] = field(default=None)
    neuron_volcano_file: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        # 新增：读取灾害类型
        disaster_type = sources.get("disaster_type", "unknown")

        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
            # 新增：把灾害类型传入batch
            data_dict['disaster_type'] = disaster_type
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['disaster_type'] = disaster_type
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # 新增：传递灾害类型
        if 'disaster_type' in instances[0]:
            batch['disaster_types'] = [inst['disaster_type'] for inst in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load_neurons_simple(file_path):
    """统一加载神经元文件工具函数"""
    if not file_path or not os.path.exists(file_path):
        return []

    neurons = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            pos_part = parts[0]
            score = float(parts[1])

            if '_' in pos_part:
                l, n = pos_part.split('_')
            elif ',' in pos_part:
                l, n = pos_part.split(',')
            else:
                continue

            try:
                neurons.append((int(l), int(n), score))
            except:
                continue

    unique = {}
    for l, n, s in neurons:
        unique[(l, n)] = s
    return [(l, n, s) for (l, n), s in unique.items()]


def build_neuron_weights(model, neuron_list):
    """根据神经元列表构建"沿 down_proj 输入维（intermediate）"的软掩码权重向量。
    自动探测参数名前缀（兼容 model.layers.X / model.model.layers.X / module.model.layers.X 等），
    ZeRO-3 下用 GatheredParameters 读完整形状。
    返回 { param_name: float tensor[intermediate] }。
    """
    if not neuron_list:
        return {}

    import re
    # 1) 找到所有 *.layers.{idx}.mlp.down_proj.weight 参数
    layer_to_param = {}
    for name, param in model.named_parameters():
        m = re.search(r'\.layers\.(\d+)\.mlp\.down_proj\.weight$', name)
        if m:
            layer_to_param[int(m.group(1))] = (name, param)

    if not layer_to_param:
        print("⚠️ [build_neuron_weights] 未找到任何 *layers.*.mlp.down_proj.weight 参数！")
        for n, _ in model.named_parameters():
            if "down_proj" in n:
                print(f"   示例参数名: {n}")
                break
        return {}

    # 2) 按 layer_idx 聚合神经元分数
    layer_to_scores = {}
    for layer_idx, neuron_pos, score_S in neuron_list:
        layer_to_scores.setdefault(layer_idx, []).append((neuron_pos, score_S))

    # 3) 用 GatheredParameters 读完整形状（ZeRO-3 下 param.shape 可能是 [0]）
    try:
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
        _has_gather = True
    except Exception:
        _has_gather = False

    _params_list = list(model._neuron_params.values()) if hasattr(model, "_neuron_params") else \
                   [p for _, p in layer_to_param.values()]
    _need_gather = _has_gather and any(hasattr(p, "ds_id") for p in _params_list)

    weights = {}
    ctx = GatheredParameters(_params_list, modifier_rank=None) if _need_gather else torch.no_grad()
    with ctx:
        for layer_idx, neuron_score_list in layer_to_scores.items():
            if layer_idx not in layer_to_param:
                print(f"⚠️ 模型中没有 layer {layer_idx} 的 down_proj，跳过")
                continue
            name, param = layer_to_param[layer_idx]

            # 此时 param 已被 all-gather，shape 应为 [hidden, intermediate]
            shape = tuple(param.shape)
            if len(shape) != 2:
                print(f"⚠️ layer {layer_idx} 参数维度异常: {name} shape={shape}，跳过")
                continue
            intermediate_size = shape[1]

            row_w = torch.zeros(intermediate_size, dtype=torch.float32)
            invalid = []
            for n, s in neuron_score_list:
                if 0 <= n < intermediate_size:
                    if float(s) > float(row_w[n]):
                        row_w[n] = float(s)
                else:
                    invalid.append(n)
            if invalid:
                print(f"⚠️ layer {layer_idx} 神经元位置无效: {invalid} "
                      f"(有效范围 0..{intermediate_size-1})")
            weights[name] = row_w

    return weights


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    # ==================== 加载所有灾害类型的神经元（预加载） ====================
    rank0_print("\n========== 预加载各灾害神经元文件 ==========")
    disaster_neuron_map = {
        "fire": load_neurons_simple(model_args.neuron_fire_file),
        "flood": load_neurons_simple(model_args.neuron_flood_file),
        "landslide": load_neurons_simple(model_args.neuron_landslide_file),
        "volcano": load_neurons_simple(model_args.neuron_volcano_file),
    }
    for typ, ns in disaster_neuron_map.items():
        rank0_print(f"✅ {typ}: {len(ns)} 个神经元")

    # ==================== 主干冻结 + 动态软掩码系统（DeepSpeed 兼容） ====================
    if model_args.freeze_backbone:
        from transformers import Trainer
        from collections import Counter
        old_trainer_training_step = Trainer.training_step
        rank0_print("\n冻结模型全部主干参数，仅动态开放灾害专属神经元（软掩码加权更新）...")
        for param in model.parameters():
            param.requires_grad = False

        # 统一解冻所有 down_proj
        target_param_names = []
        for name, param in model.named_parameters():
            if "mlp.down_proj.weight" in name:
                param.requires_grad = True
                target_param_names.append(name)
        rank0_print(f"✅ 统一解冻 {len(target_param_names)} 个down_proj参数")

        # 登记需要控制的参数
        model._neuron_params = {}
        for name, param in model.named_parameters():
            if name in target_param_names:
                model._neuron_params[name] = param

        # 当前 batch 生效的行级软权重向量：{ param_name: float[I] }
        model._current_masks = {}
        # 训练全程初始快照（在第一次 training_step 时填充，见 new_training_step）
        model._neuron_init_snapshot = {}
        # 软掩码抽检标记（只打印一次）
        model._soft_mask_checked = False
        # DeepSpeed 兼容：每次 step 前的快照 + pending 标记
        model._neuron_step_snapshot = {}
        model._pending_soft_mask = False

        rank0_print("✅ 神经元参数已登记，将采用 DeepSpeed 兼容的软掩码加权更新。")

        # ---------------- 核心：在 training_step 开头做"上一步"的加权写回 ----------------
        def _apply_soft_mask_after_step(model):
            """在 DeepSpeed 已执行上一步 optimizer.step() 后做软掩码加权写回。
               model 是 base_model（原始模型，不是 DeepSpeedEngine）。
               ZeRO-3 + offload 单卡下完整参数在 param.ds_tensor.data（扁平化分片），用 ds_shape 还原。
            """
            with torch.no_grad():
                for name, param in model._neuron_params.items():
                    if name not in model._neuron_step_snapshot:
                        continue
                    snap_cpu = model._neuron_step_snapshot[name]

                    # 取完整参数（CPU 侧）
                    if hasattr(param, "ds_tensor"):
                        ds_data = param.ds_tensor.data
                        ds_data = ds_data.cpu() if ds_data.device.type != "cpu" else ds_data
                        if ds_data.numel() == param.ds_numel:
                            full_param = ds_data.view(param.ds_shape)
                        else:
                            if not getattr(model, "_partition_warned", False):
                                rank0_print(f"[诊断] {name} 分片不完整 "
                                            f"(numel={ds_data.numel()} < ds_numel={param.ds_numel})，跳过")
                                model._partition_warned = True
                            continue
                    else:
                        full_param = param.data.cpu()

                    if snap_cpu.shape != full_param.shape:
                        if not getattr(model, "_shape_mismatch_warned", False):
                            rank0_print(f"[诊断] {name} shape 不匹配: "
                                        f"snap={tuple(snap_cpu.shape)}, full={tuple(full_param.shape)}")
                            model._shape_mismatch_warned = True
                        continue

                    row_w = model._current_masks.get(name, None)
                    if row_w is None:
                        row_w = torch.zeros(full_param.shape[1], dtype=torch.float32)
                    else:
                        row_w = row_w.float()

                    full_w = row_w.unsqueeze(0).expand_as(full_param)
                    unsel = (full_w == 0)

                    if unsel.any():
                        model._last_unselected_delta_before = (
                            full_param[unsel].float() - snap_cpu[unsel].float()
                        ).abs().max().item()

                    delta = (full_param - snap_cpu).to(torch.float32)
                    # ===== 诊断：第一次执行时打印选中行 delta 的量级 =====
                    if not getattr(model, "_grad_check_done", False):
                        sel_mask = full_w > 0
                        if sel_mask.any():
                            sel_delta = delta[sel_mask]
                            rank0_print(
                                f"[诊断] {name} 选中行 delta: "
                                f"max={sel_delta.abs().max().item():.3e}, "
                                f"mean={sel_delta.abs().mean().item():.3e}, "
                                f"未选中行 before={getattr(model, '_last_unselected_delta_before', 0):.3e}"
                            )
                        else:
                            rank0_print(f"[诊断] {name} 无选中行")
                        model._grad_check_done = True
                    # ===== 诊断结束 =====
                    new_w = snap_cpu.float() + full_w * delta

                    if hasattr(param, "ds_tensor"):
                        flat_new = new_w.reshape(-1).to(param.ds_tensor.data.dtype)
                        if param.ds_tensor.data.numel() == flat_new.numel():
                            param.ds_tensor.data.copy_(flat_new)
                        else:
                            continue
                    else:
                        param.data.copy_(new_w.to(param.data.dtype))

                    if unsel.any():
                        if hasattr(param, "ds_tensor"):
                            after_cpu = param.ds_tensor.data.cpu()
                            if after_cpu.numel() == param.ds_numel:
                                after_cpu = after_cpu.view(param.ds_shape)
                            else:
                                after_cpu = None
                        else:
                            after_cpu = param.data.cpu()
                        if after_cpu is not None and after_cpu.shape == snap_cpu.shape:
                            model._last_unselected_delta_after = (
                                after_cpu[unsel].float() - snap_cpu[unsel].float()
                            ).abs().max().item()

                    del delta, new_w, full_w, unsel

                    if not model._soft_mask_checked:
                        sel_idx = torch.nonzero(row_w > 0).flatten()[:3]
                        for idx in sel_idx.tolist():
                            d = (full_param[:, idx].float() - snap_cpu[:, idx].float()).abs().max().item()
                            if d > 0:
                                rank0_print(f"[软掩码抽检] {name} col={idx} "
                                            f"s={row_w[idx].item():.6f}, |Δ|={d:.3e}")
                        model._soft_mask_checked = True

        def new_training_step(self, model, inputs):
            # DeepSpeed 包装后 model 是 DeepSpeedEngine，原始模型在 model.module
            base_model = model.module if hasattr(model, "module") else model

            # ---------- 1. 如果上一步的 AdamW 更新已发生，先做软掩码写回 ----------
            if getattr(base_model, "_pending_soft_mask", False) and base_model._neuron_step_snapshot:
                _apply_soft_mask_after_step(base_model)
                base_model._pending_soft_mask = False

            # ---------- 2. 切换当前 batch 的灾害类型与权重 ----------
            disaster_types = inputs.pop("disaster_types", [])
            if len(disaster_types) > 0:
                main_type = Counter(disaster_types).most_common(1)[0][0]
            else:
                main_type = "unknown"

            neurons = disaster_neuron_map.get(main_type, [])
            # build_neuron_weights 传入 base_model，保证能拿到 _neuron_params
            base_model._current_masks = build_neuron_weights(base_model, neurons)
            base_model._current_disaster_type = main_type
            n_active = sum(int((w > 0).sum().item()) for w in base_model._current_masks.values())
            rank0_print(f"🔄 当前batch灾害类型: {main_type} | 选中神经元数量: {n_active}")

            # ---------- 3. 为本次 step 保存 step 前快照（CPU 保存） ----------
            _snap = {}
            for _n, _p in base_model._neuron_params.items():
                if hasattr(_p, "ds_tensor"):
                    _ds = _p.ds_tensor.data
                    _ds = _ds.cpu() if _ds.device.type != "cpu" else _ds
                    if _ds.numel() == _p.ds_numel:
                        _snap[_n] = _ds.view(_p.ds_shape).detach().clone()
                    else:
                        if not getattr(base_model, "_snap_partition_warned", False):
                            rank0_print(f"[诊断] {_n} 快照分片不完整 "
                                        f"(numel={_ds.numel()} < ds_numel={_p.ds_numel})，跳过")
                            base_model._snap_partition_warned = True
                        continue
                else:
                    _snap[_n] = _p.data.detach().to("cpu", copy=True)
            base_model._neuron_step_snapshot = _snap

            # 第一次调用时保存"训练全程初始快照"
            if not base_model._neuron_init_snapshot:
                base_model._neuron_init_snapshot = {
                    _n: _v.clone() for _n, _v in _snap.items()
                }
                rank0_print("✅ 已保存训练全程初始快照（用于验证未选中行严格冻结）")

            base_model._pending_soft_mask = True

            # ---------- 4. 正常前向+反向，DeepSpeed 会在返回后执行 step ----------
            loss = old_trainer_training_step(self, model, inputs)
            return loss

        Trainer.training_step = new_training_step

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    model.get_input_embeddings().requires_grad_(True)
    # 打印可训练参数
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"\n===== 参数统计 =====")
    rank0_print(f"总参数: {total:,}")
    rank0_print(f"可训练: {trainable:,} ({trainable/total*100:.2f}%)")

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    # ========== 强制禁用数据打乱 ==========
    from torch.utils.data import SequentialSampler
    trainer._get_train_sampler = lambda: SequentialSampler(trainer.train_dataset)
    rank0_print("✅ 已禁用数据打乱，使用顺序采样器")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # ========== 训练结束后：对最后一次 step 补做软掩码写回 ==========
    if model_args.freeze_backbone and getattr(model, "_pending_soft_mask", False):
        _apply_soft_mask_after_step(model)
        model._pending_soft_mask = False

    # ========== 冻结/软掩码验证日志（回复审稿人） ==========
    if model_args.freeze_backbone and getattr(model, "_neuron_params", None):
        rank0_print("\n================ 未选中行冻结 & 选中行软更新验证 ================")
        rank0_print(f"[诊断] _neuron_params={len(model._neuron_params)} 个, "
                    f"_neuron_init_snapshot={len(model._neuron_init_snapshot)} 个, "
                    f"_neuron_step_snapshot={len(model._neuron_step_snapshot)} 个")
        if model._neuron_params:
            rank0_print(f"[诊断] _neuron_params[0]={list(model._neuron_params.keys())[0]}")
        if model._neuron_init_snapshot:
            rank0_print(f"[诊断] _neuron_init_snapshot[0]={list(model._neuron_init_snapshot.keys())[0]}")
        _before = getattr(model, "_last_unselected_delta_before", None)
        _after = getattr(model, "_last_unselected_delta_after", None)
        if _before is None:
            rank0_print(f"[step级抽检] 未选中行 |Δ|: 未记录（_apply_soft_mask_after_step 未被触发或全部跳过）")
        else:
            rank0_print(f"[step级抽检] 最近一步未选中行 |Δ|: "
                        f"恢复前(AdamW漂移)={_before:.3e}, "
                        f"恢复后(应为0)={_after:.3e}")

        # ZeRO-3 下必须 all-gather 后才能与完整 init 比较
        # ZeRO-3 + offload 单卡下完整参数在 ds_tensor.data（CPU），直接比较
        all_zero = True
        with torch.no_grad():
            for name, param in model._neuron_params.items():
                if name not in model._neuron_init_snapshot:
                    continue

                # 取完整参数（CPU 侧）
                # 单卡 world_size=1 时它包含全部元素，可用 ds_shape 还原。
                if hasattr(param, "ds_tensor"):
                    ds_data = param.ds_tensor.data
                    ds_data = ds_data.cpu() if ds_data.device.type != "cpu" else ds_data
                    if ds_data.numel() == param.ds_numel:
                        full_param = ds_data.view(param.ds_shape)
                    else:
                        # 分片不完整（多卡场景），无法在单卡重建完整参数，跳过
                        if not getattr(model, "_partition_warned", False):
                            rank0_print(f"[诊断] {name} 分片不完整 "
                                        f"(numel={ds_data.numel()} < ds_numel={param.ds_numel})，跳过")
                            model._partition_warned = True
                        continue
                else:
                    full_param = param.data.cpu()

                init = model._neuron_init_snapshot[name].cpu()
                if init.shape != full_param.shape:
                    rank0_print(f"[{name}] 形状不匹配 (full={tuple(full_param.shape)}, init={tuple(init.shape)})，跳过")
                    continue

                row_w = model._current_masks.get(name, None)
                if row_w is None:
                    row_w = torch.zeros(full_param.shape[1], dtype=torch.float32)
                else:
                    row_w = row_w.float()
                full_w = row_w.unsqueeze(0).expand_as(full_param)

                diff = (full_param.float() - init.float()).abs()
                unsel = full_w == 0
                sel = full_w > 0
                unsel_max = diff[unsel].max().item() if unsel.any() else 0.0
                sel_min = diff[sel].min().item() if sel.any() else float("nan")
                sel_max = diff[sel].max().item() if sel.any() else float("nan")
                rank0_print(f"[{name}] 全程未选中行 |Δ|max = {unsel_max:.3e} | "
                            f"选中行 |Δ| ∈ [{sel_min:.3e}, {sel_max:.3e}]")
                if unsel_max != 0.0:
                    all_zero = False
        rank0_print(f"结论: 训练全程未选中行参数前后差 "
                    f"{'全部严格为 0 ✅' if all_zero else '存在非零 ❌'}")
        rank0_print("=================================================================\n")

    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()