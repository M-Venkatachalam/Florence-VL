import os
from .clip_encoder import CLIPVisionTower
from .florence_encoder import FlorenceVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    print('loading vision tower {}'.format(vision_tower))

    is_absolute_path_exists = os.path.exists(vision_tower)

    if "openai" in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    return FlorenceVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
