from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torch
import psutil
class Logger:
    def __init__(self, hps):
        self.hps = hps 
        self.log_dir = (hps['log_dir']/ hps['envname']).resolve()
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.saved_episode_dir = self.log_dir / 'episodes'
        self.saved_episode_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.summary_writter = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard_events'))
        self.current_process = psutil.Process()
        return 
    def save_episodes(self, episodes):
        torch.save(episodes, str(self.saved_episode_dir))
        return 
    def save_checkpoint(self, model, i_eposide):
        saved_dict = {
            'policynet': model.policy_net.state_dict(), 
            'targetnet': model.target_net.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }
        torch.save(saved_dict, str(self.checkpoint_dir / str('saved_model-' + str(i_eposide) + '.pth')))
        return 
    def add_sys_info(self, i_eposide):
        self.summary_writter.add_scalar('sys/cpu_percentage', self.current_process.cpu_percent(), i_eposide)
        self.summary_writter.add_scalar(
            'sys/memory_percent', self.current_process.memory_percent(), i_eposide)
        return 
    def load_checkpoint(self, path):

        return
    def add_scalars(self, *args, **kwargs):
        return self.summary_writter.add_scalars(*args, **kwargs)
    def add_scalar(self, *args, **kwargs):
        return self.summary_writter.add_scalar(*args, **kwargs)
    def add_graph(self, model, **kwargs):
        self.summary_writter.add_graph(model.policy_net, **kwargs)
        self.summary_writter.add_graph(model.target_net, **kwargs)
        return
    def close(self):
        return self.summary_writter.close()
    
