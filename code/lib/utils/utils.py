from enum import Enum
from threading import Thread
import torch
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
import subprocess
import time
def setup(rank, world_size, port='12357'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    
    os.environ['TORCH_NCCL_TIMEOUT'] = '1800'  
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=torch.distributed.default_pg_timeout * 3)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all processes"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class GPUTempMonitor:
    def __init__(self, temp_threshold=85, check_interval=5):
        """
        Monitor GPU temperature and control execution to maintain temperature below threshold
        
        Args:
            temp_threshold (int): Temperature threshold in Celsius
            check_interval (int): How often to check temperature in seconds
        """
        self.temp_threshold = temp_threshold
        self.check_interval = check_interval
        self.stop_monitoring = False
        self.pause_execution = False
        self.monitor_thread = None
        
    def get_gpu_temp(self):
        """Get current GPU temperature using nvidia-smi"""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            # Parse multiple GPU temperatures
            temps = [int(x.strip()) for x in result.strip().split('\n')]
            return temps
        except Exception as e:
            print(f"Error getting GPU temperature: {e}")
            return [0]  # Return 0 if can't get temperature
    
    def monitor_temperature(self):
        """Monitor loop that checks GPU temperature and sets pause flag"""
        while not self.stop_monitoring:
            temps = self.get_gpu_temp()
            max_temp = max(temps)
            
            if max_temp >= self.temp_threshold:
                if not self.pause_execution:
                    print(f"\nGPU temperature ({max_temp}°C) exceeded threshold ({self.temp_threshold}°C). Pausing training...")
                    self.pause_execution = True
            elif self.pause_execution and max_temp < self.temp_threshold - 5:  # 5°C hysteresis
                print(f"\nGPU temperature ({max_temp}°C) cooled down. Resuming training...")
                self.pause_execution = False
            
            # Sleep for check interval
            time.sleep(self.check_interval)
    
    def start(self):
        """Start the temperature monitoring thread"""
        self.monitor_thread = Thread(target=self.monitor_temperature, daemon=True)
        self.monitor_thread.start()
        print(f"GPU temperature monitoring started (threshold: {self.temp_threshold}°C)")
        
    def stop(self):
        """Stop the temperature monitoring thread"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
