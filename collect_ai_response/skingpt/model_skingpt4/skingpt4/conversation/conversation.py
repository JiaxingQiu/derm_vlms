import argparse
import time
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any, Optional, Dict
import json
import math

from skingpt4.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    # system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    @torch.no_grad()
    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=0.0, max_length=2000, print_prompt=False, train_mode=True,
               label_words: Optional[List[str]] = None):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb(conv, img_list, print_prompt)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        do_sample = False if temperature < 1e-4 else True
        temperature = 0.0 if temperature < 1e-4 else temperature
        outputs = self.model.llm_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llm_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text

        label_losses = None
        label_losses_str = ""
        if label_words:
            # Compute cross-entropy for each candidate continuation given the same context
            label_losses = self._score_label_words(embs, label_words)
            # format label_losses into a string 
            label_losses_str = json.dumps(label_losses, separators=(',', ':'), sort_keys=True)

        response_dict = {
            "output_text": output_text,
            "output_token": output_token.cpu().numpy(),
            "label_losses": label_losses,
            "label_losses_str": label_losses_str,
        }
        return response_dict

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list, print_prompt=False):
        prompt = conv.get_prompt()
        if print_prompt: # Print the exact text prompt used around image placeholders for debugging
            print("=== Prompt ===")
            print(prompt)
            print("==============")
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llm_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llm_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def _nll_to_probs(self, label_losses: Dict[str, Dict[str, float]], use_avg: bool = False, temperature: float = 1.0) -> Dict[str, float]:
        scores = {}
        temp = max(1e-8, float(temperature))
        for lbl, d in label_losses.items():
            nll = float(d['avg_nll'] if use_avg else d['sum_nll'])
            scores[lbl] = -(nll) / temp
        m = max(scores.values()) if scores else 0.0
        exps = {lbl: math.exp(s - m) for lbl, s in scores.items()}
        Z = sum(exps.values()) or 1.0
        return {lbl: v / Z for lbl, v in exps.items()}

    def _score_label_words(self, context_embs: torch.Tensor, label_words: List[str], use_prob: bool = True, use_avg: bool = False, temperature: float = 1.0) -> Dict[str, Dict[str, float]]:
        # Scores each label word by negative log-likelihood (sum and average over its tokens)
        results = {}
        context_len = context_embs.shape[1]
        for word in label_words:
            tokenized = self.model.llm_tokenizer(
                " " + word,
                return_tensors="pt",
                add_special_tokens=False,
            )
            cand_ids = tokenized.input_ids.to(self.device)  # shape: [1, T]
            # Embed candidate tokens
            cand_embs = self.model.llm_model.model.embed_tokens(cand_ids)
            # Build full embeddings: context + candidate
            full_embs = torch.cat([context_embs, cand_embs], dim=1)
            # Labels: ignore context, supervise candidate tokens
            labels = torch.full(
                (1, full_embs.shape[1]),
                -100,
                dtype=torch.long,
                device=self.device,
            )
            labels[0, context_len:context_len + cand_ids.shape[1]] = cand_ids[0]
            outputs = self.model.llm_model(
                inputs_embeds=full_embs,
                labels=labels,
                return_dict=True,
            )
            avg_nll = float(outputs.loss.item())  # mean over candidate tokens
            num_tokens = int(cand_ids.shape[1])
            sum_nll = avg_nll * num_tokens
            results[word] = {"sum_nll": sum_nll,
                             "avg_nll": avg_nll,
                             "num_tokens": float(num_tokens)}
        if use_prob and results:
            probs = self._nll_to_probs(results, use_avg=use_avg, temperature=temperature)
            for lbl, p in probs.items():
                if lbl in results:
                    results[lbl]["prob"] = float(p)
        return results




# class Chat:
#     def __init__(self, model, vis_processor, device='cuda:0'):
#         self.device = device
#         self.model = model
#         self.vis_processor = vis_processor
#         stop_words_ids = [torch.tensor([835]).to(self.device),
#                           torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
#         self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

#     def ask(self, text, conv):
#         if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
#                 and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
#             conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
#         else:
#             conv.append_message(conv.roles[0], text)

#     def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
#                repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, print_prompt=False):
        
#         self.model.eval()
#         conv.append_message(conv.roles[1], None)
#         with torch.no_grad():
#             embs = self.get_context_emb(conv, img_list, print_prompt)
#             current_max_len = embs.shape[1] + max_new_tokens
#             if current_max_len - max_length > 0:
#                 print('Warning: The number of tokens in current conversation exceeds the max length. '
#                     'The model will not see the contexts outside the range.')
#             begin_idx = max(0, current_max_len - max_length)
#             embs = embs[:, begin_idx:]
#             do_sample = False if temperature < 1e-4 else True
#             temperature = 0.0 if temperature < 1e-4 else temperature
#             outputs = self.model.llm_model.generate(
#                 inputs_embeds=embs,
#                 max_new_tokens=max_new_tokens,
#                 stopping_criteria=self.stopping_criteria,
#                 num_beams=num_beams,
#                 do_sample=do_sample,
#                 min_length=min_length,
#                 top_p=top_p,
#                 repetition_penalty=repetition_penalty,
#                 length_penalty=length_penalty,
#                 temperature=temperature,
#             )
#             output_token = outputs[0]
#             if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
#                 output_token = output_token[1:]
#             if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
#                 output_token = output_token[1:]
#             output_text = self.model.llm_tokenizer.decode(output_token, add_special_tokens=False)
#             output_text = output_text.split('###')[0]  # remove the stop sign '###'
#             output_text = output_text.split('Assistant:')[-1].strip()
#             conv.messages[-1][1] = output_text
#         return output_text, output_token.cpu().numpy()

#     def upload_img(self, image, conv, img_list):
#         self.model.eval()
#         if isinstance(image, str):  # is a image path
#             raw_image = Image.open(image).convert('RGB')
#             image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
#         elif isinstance(image, Image.Image):
#             raw_image = image
#             image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
#         elif isinstance(image, torch.Tensor):
#             if len(image.shape) == 3:
#                 image = image.unsqueeze(0)
#             image = image.to(self.device)
#         with torch.no_grad():
#             image_emb, _ = self.model.encode_img(image)
#         img_list.append(image_emb)
#         conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
#         msg = "Received."
#         # self.conv.append_message(self.conv.roles[1], msg)
#         return msg

#     def get_context_emb(self, conv, img_list, print_prompt=False):
#         self.model.eval()
#         with torch.no_grad():
#             prompt = conv.get_prompt()
#             if print_prompt: # Print the exact text prompt used around image placeholders for debugging
#                 print("=== Prompt ===")
#                 print(prompt)
#                 print("==============")
#             prompt_segs = prompt.split('<ImageHere>')
#             assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
#             seg_tokens = [
#                 self.model.llm_tokenizer(
#                     seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
#                 # only add bos to the first seg
#                 for i, seg in enumerate(prompt_segs)
#             ]
#             seg_embs = [self.model.llm_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
#             mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
#             mixed_embs = torch.cat(mixed_embs, dim=1)
#         return mixed_embs


