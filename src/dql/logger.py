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
        self.episode_offset = 0
        return 
    def save_episodes(self, episodes):
        torch.save(episodes, str(self.saved_episode_dir))
        return 
    def add_episode_offset(self, current_episode):
        return self.episode_offset + current_episode
    def set_episode_offset(self, offset):
        self.episode_offset = offset
        return 
    def save_checkpoint(self, model, i_episode):
        # print(self.add_episode_offset(i_episode))
        saved_dict = {
            'policynet': model.policy_net.state_dict(), 
            'optimizer': model.optimizer.state_dict()
        }
        torch.save(saved_dict, str(self.checkpoint_dir / 'saved_model.ckpt-{}'.format(self.add_episode_offset(i_episode))))
        return 
    def add_sys_info(self, i_episode):
        self.summary_writter.add_scalar('sys/cpu_percentage', self.current_process.cpu_percent(), self.add_episode_offset(i_episode))
        self.summary_writter.add_scalar(
            'sys/memory_percent', self.current_process.memory_percent(), self.add_episode_offset(i_episode))
        return 
    def add_scalar(self, tag, scalar_value, i_episode):
        return self.summary_writter.add_scalar(tag, scalar_value, global_step=self.add_episode_offset(i_episode))
    def add_graph(self, model, **kwargs):
        if self.episode_offset != 0:
            return 
        self.summary_writter.add_graph(model.policy_net, **kwargs)
        self.summary_writter.add_graph(model.target_net, **kwargs)
        return
    def close(self):
        return self.summary_writter.close()
    
