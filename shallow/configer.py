from typing import Dict, List, Any
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class _BaseTransformers: AUG: str = ''


@dataclass
class Transformers:
    """common parameters for transform pipelines"""

    TRAIN: _BaseTransformers = _BaseTransformers(AUG='train')
    VALID: _BaseTransformers = _BaseTransformers(AUG='valid')
    TEST: _BaseTransformers = _BaseTransformers(AUG='test')
    UNSUPERVISED: _BaseTransformers = _BaseTransformers(AUG='unsupervised')

    MEAN: List[float] = field(default_factory=lambda: [0,0,0])
    STD: List[float] =  field(default_factory=lambda: [1,1,1])

    CROP: List[int] = field(default_factory=lambda: [128, 128])
    CROP_VAL: List[int] = field(default_factory=lambda: [256, 256])
    RESIZE: List[int] = field(default_factory=lambda: [128, 128])

    WORKERS: int = 1


@dataclass
class _BaseData:
    DATASETS: List[str] = field(default_factory=lambda: [])
    SPLITS: Dict[str, Any] = field(default_factory=lambda: dict())
    GPU_PRELOAD: bool = False
    PRELOAD: bool = False
    MULTIPLY: Dict[str, Any] = field(default_factory=lambda: dict(rate=0))


@dataclass
class Data:
    TRAIN: _BaseData = _BaseData()
    VALID: _BaseData = _BaseData()
    TEST : _BaseData = _BaseData()
    UNSUPERVISED : _BaseData = _BaseData()


@dataclass
class LoggerConfig:
    # TODO
    pass


@dataclass
class Parallel:
    DDP: bool = False
    LOCAL_RANK: int = -1
    WORLD_SIZE: int = 0
    IS_MASTER: bool = False


@dataclass
class Hydra_Opt:
    # TODO: how?
    # sweep dir
    # mutlirun dir
    run: Dict[str, Any] = field(default_factory=lambda: dict(dir='./output/${now:%m-%d}/${now:%H-%M-%S}'))


@dataclass
class Train:
    AMP: bool = False
    GPUS: List[float] = field(default_factory=lambda: [0,])
    NUM_WORKERS: int = 1
    EPOCH_COUNT: int = 0
    BATCH_SIZE: int = 0
    NUM_SPLITS: int = 0
    SPLIT_ID: str = ''
    SAVE_STEP: float = 1.
    SCALAR_STEP: int = 1
    TB_STEP: int = 1
    INIT_MODEL: str = ''
    INIT_ENCODER: str = ''
    EARLY_EXIT: int = 10
    SEED: int = 0


@dataclass
class Valid:
    STEP: int = '${TRAIN.TB_STEP}'
    BATCH_SIZE: int = '${TRAIN.BATCH_SIZE}'


@dataclass
class Test:
    BATCH_SIZE: int = '${TRAIN.BATCH_SIZE}'


@dataclass
class Features:
    CLAMP: float = 1000.
    TENSOR_DATASET: bool = False
    BATCH_ACCUMULATION_STEP: int = 1
    GRAD_CLIP: float = 10.0
    FREEZE_ENCODER: float = .0


def _generate_node(group, name, node_class, **kwargs):
    return dict(group=group, name=name, node=node_class, **kwargs)


def example_generate_default_nodes():
    nodes = [
        _generate_node(group='TRANSFORMERS', name="_transformers", node_class=Transformers),
        _generate_node(group='DATA', name="_data", node_class=Data),
        _generate_node(group='FEATURES', name="_features", node_class=Features),
        _generate_node(group='PARALLEL', name="_parallel", node_class=Parallel),
        _generate_node(group='TRAIN', name="_train", node_class=Train),
        _generate_node(group='VALID', name="_valid", node_class=Valid),
        _generate_node(group='TEST', name="_test", node_class=Test),
    ]
    return nodes


def cfg_init(get_nodes_fn, **kwargs):
    #node = dict(group=GROUP, name=NAME, node=NODE)
    nodes = get_nodes_fn(**kwargs)
    cs = ConfigStore.instance()
    for node in nodes:
        cs.store(**node)


# @hydra.main(config_path="configs", config_name="base")
# def main(cfg) -> None:
#     # Test cfg run
#     print(OmegaConf.to_yaml(cfg, resolve=True))
#     return cfg


# if __name__ == '__main__':
#     cfg_init()
#     main()
