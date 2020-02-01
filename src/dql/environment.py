
import gym
import torch
from torchvision import transforms as T
from PIL import Image
class ScreenProcessor:
    def __init__(self, env, image_size):
        self.env = env
        self.transform = T.Compose([T.ToPILImage(),
                                    T.Resize(
                                        image_size, interpolation=Image.CUBIC),
                                    T.functional.to_grayscale,
                                    T.ToTensor()])
        return

    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        return self.transform(screen)
class Environment:
    def __init__(self, hps):
        self.hps = hps 
        self.env = gym.make(hps['envname'])
        self.screen_processor = ScreenProcessor(self.env, self.hps['image_size'])
        self.action_space = self.env.action_space
        return
    def get_init_state(self):
        self.env.reset()
        current_screen = self.screen_processor.get_screen()
        state = torch.cat([current_screen] * self.hps['frame_length'], dim=0)
        return state
    def get_screen(self):
        return self.screen_processor.get_screen()
    def step(self, *args, **kwargs):
        return self.env.step(*args, **kwargs)
    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)