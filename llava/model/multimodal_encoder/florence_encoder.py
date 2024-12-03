import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, AutoProcessor, AutoModelForCausalLM 




class FlorenceVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.load_model()


    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = AutoProcessor.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        self.vision_tower = AutoModelForCausalLM.from_pretrained(self.vision_tower_name, trust_remote_code=True).to(torch.bfloat16)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True


    @torch.no_grad()
    def forward(self, images):
        
        
        # task = [
        #     'Describe in detail what is shown in the image.',
        #     'What is the text in the image?',
        #     'Locate the objects in the image, with their descriptions.',
        # ]

        ## token id for three tasks prompt
        task_ids = torch.tensor([
            [0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2, 1],
            [0, 2264, 16, 5, 2788, 11, 5, 2274, 116, 2, 1, 1, 1, 1],
            [0, 574, 22486, 5, 8720, 11, 5, 2274, 6, 19, 49, 24173, 4, 2]
        ]).to(device=self.device)


        with torch.no_grad():
            generated_ids, image_feature, encoder_last_hidden_state = self.vision_tower.generate(
                input_ids=task_ids,
                pixel_values=images,
                max_new_tokens=1,
                do_sample=False,
                num_beams=1,
            )
            return image_feature, encoder_last_hidden_state


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
