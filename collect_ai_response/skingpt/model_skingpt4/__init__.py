import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

# Ensure 'skingpt4' (inner package) is importable: add its parent (this dir) to sys.path
_PKG_DIR = os.path.dirname(__file__)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


# Absolute imports now resolve to inner package
from skingpt4.common.config import Config
from skingpt4.common.dist_utils import get_rank
from skingpt4.common.registry import registry
from skingpt4.conversation.conversation import Chat, CONV_VISION

# register modules for side-effects (builders/models/processors/runners/tasks)
from skingpt4.datasets.builders import *  # noqa
from skingpt4.models import *  # noqa
from skingpt4.processors import *  # noqa
from skingpt4.runners import *  # noqa
from skingpt4.tasks import *  # noqa


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def init_cfg(gpu_id = 0, cfg_path_str = None):
    # robust default cfg path
    if cfg_path_str is None:
        cfg_path_str = "skingpt4_eval_llama2_13bchat"
    cfg_path = os.path.join(_PKG_DIR, "eval_configs", f"{cfg_path_str}.yaml")
    class Args:
        def __init__(self, cfg_path, gpu_id):
            self.cfg_path = cfg_path
            self.gpu_id = gpu_id
            self.options = None
    print(f"Initializing Configs")
    args = Args(cfg_path, gpu_id)
    cfg = Config(args)
    cfg.gpu_id = gpu_id
    return cfg


def init_chat(cfg):
    print("Initializing Chat")
    model_config = cfg.model_cfg
    model_config.device_8bit = cfg.gpu_id  # used when low_resource=True
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f"cuda:{cfg.gpu_id}")
    model.ckpt = cfg.config.model.ckpt # expose the ckpt path

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device=f"cuda:{cfg.gpu_id}")
    print("Initialization Finished")
    return model, vis_processor, chat


def load_model_weights(model, model_path: str):
    state = torch.load(model_path, map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    return model


def chat_with_image(chat, image, question, num_beams=1, temperature=0.01, remove_system=True, print_prompt=False, train_mode=True,
                    label_words=["malignant", "benign", "other"]):
    chat_state = CONV_VISION.copy()
    if remove_system:
        chat_state.system = ""  # mirror training: no system line
    img_list = []
    _ = chat.upload_img(image, chat_state, img_list)
    chat.ask(question, chat_state)
    response_dict = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=500,
        print_prompt=print_prompt,
        train_mode=train_mode,
        label_words=label_words,
    )
    response = response_dict["output_text"]
    response += "\n###NLL:" + response_dict["label_losses_str"]
    return response