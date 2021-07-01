import os
import yaml
import torch
from utils.parser import edict2dict
from easydict import EasyDict as edict


def compute_edge_ratio(G_list):
  num_edges_max, num_edges = .0, .0
  for gg in G_list:
    num_nodes = gg.number_of_nodes()
    num_edges += gg.number_of_edges()
    num_edges_max += num_nodes**2

  ratio = (num_edges_max - num_edges) / num_edges
  return ratio


def data_to_gpu(*input_data):
  return_data = []
  for dd in input_data:
    if type(dd).__name__ == 'Tensor':
      return_data += [dd.cuda()]
  
  return tuple(return_data)


def snapshot(model, optimizer, config, step, gpus=[0], tag=None, scheduler=None):
  
  if scheduler is not None:
    model_snapshot = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step
    }
  else:
    model_snapshot = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),        
        "step": step
    }    

  torch.save(model_snapshot,
             os.path.join(config.save_dir, "model_snapshot_{}.pth".format(tag)
                          if tag is not None else
                          "model_snapshot_{:07d}.pth".format(step)))


def load_model(model, file_name, device, optimizer=None, scheduler=None):
  model_snapshot = torch.load(file_name, map_location=device)  
  model.load_state_dict(model_snapshot["model"])
  if optimizer is not None:
    optimizer.load_state_dict(model_snapshot["optimizer"])

  if scheduler is not None:
    scheduler.load_state_dict(model_snapshot["scheduler"])


class EarlyStopper(object):
  """ 
    Check whether the early stop condition (always 
    observing decrease in a window of time steps) is met.

    Usage:
      my_stopper = EarlyStopper([0, 0], 1)
      is_stop = my_stopper.tick([-1,-1]) # returns True
  """

  def __init__(self, init_val, win_size=10, is_decrease=True):
    if not isinstance(init_val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    self._win_size = win_size
    self._num_val = len(init_val)
    self._val = [[False] * win_size for _ in range(self._num_val)]
    self._last_val = init_val[:]
    self._comp_func = (lambda x, y: x < y) if is_decrease else (
        lambda x, y: x >= y)

  def tick(self, val):
    if not isinstance(val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    assert len(val) == self._num_val

    for ii in range(self._num_val):
      self._val[ii].pop(0)

      if self._comp_func(val[ii], self._last_val[ii]):
        self._val[ii].append(True)
      else:
        self._val[ii].append(False)

      self._last_val[ii] = val[ii]

    is_stop = all([all(xx) for xx in self._val])

    return is_stop
