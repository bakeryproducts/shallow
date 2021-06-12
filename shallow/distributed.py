""" Distributed training/validation utils
Ross Wightman
"""
import torch
from shallow.utils.nn import unwrap_model


def gather_tensor(tensor): 
     ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1 
     tensor_list = [torch.zeros(tensor.shape, dtype=tensor.dtype, device=tensor.device) for _ in range(ws)] 
     torch.distributed.all_gather(tensor_list, tensor) 
     tensor = torch.vstack(tensor_list) 
     return tensor 
  
 def reduce_tensor(tensor): 
     rt = tensor.clone() 
     torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM) 
     rt /= ( torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1) 
     return rt 


def distribute_bn(model, reduce=False):
    # ensure every node has the same running bn stats
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1 
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=torch.distributed.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)

