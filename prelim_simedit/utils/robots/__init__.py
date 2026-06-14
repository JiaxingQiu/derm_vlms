"""Robot registry. Add a new medrobot here to make it plug-and-play."""

from .medgemma import MedGemmaRobot
from .dermato_llama import DermatoLlamaRobot

ROBOT_REGISTRY = {
    MedGemmaRobot.name: MedGemmaRobot,
    DermatoLlamaRobot.name: DermatoLlamaRobot,
}


def get_robot(name, **kwargs):
    if name not in ROBOT_REGISTRY:
        raise KeyError(f"Unknown robot '{name}'. Available: {list(ROBOT_REGISTRY)}")
    return ROBOT_REGISTRY[name](**kwargs)
