from mltool.loggingsystem import LoggingSystem
import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from data_provider.data_factory import data_provider


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    return param_sum, buffer_sum, all_size

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
        }
        #self.device = self._acquire_device()
        self.model = self._build_model()
        self.device = next(self.model.parameters()).device
        self.dataset_pool={}
    def _build_model(self):
        if self.args.model == "MICN":
            print(f"to use MICN, we ignore the label_len={self.args.label_len}. we force set label_len = {self.args.seq_len}")
            self.args.label_len = self.args.seq_len
        if self.args.model == "ETSformer":
            print(f"to use ETSformer, we ignore the d_layers={self.args.d_layers}. we force set d_layers = {self.args.e_layers}")
            self.args.d_layers = self.args.e_layers    
        model = self.model_dict[self.args.model].Model(self.args).float()
        param_sum, buffer_sum, all_size = getModelSize(model)
        print(f"Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB\n")
        #raise 
        args  = self.args
        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], find_unused_parameters=args.find_unused_parameters)
        else:
            model = model.cuda()

        # if self.args.use_multi_gpu and self.args.use_gpu:
        #     model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        if flag not in self.dataset_pool:
            self.dataset_pool[flag] = data_provider(self.args, flag)
        data_set, data_loader = self.dataset_pool[flag]
        return data_set, data_loader

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

import json
def create_logsys(args, save_config=True):
    local_rank = args.local_rank
    SAVE_PATH = args.SAVE_PATH
    logsys = LoggingSystem(local_rank == 0 or (
        not args.distributed), args.SAVE_PATH, use_wandb=args.use_wandb)
    hparam_dict = {'lr': args.learning_rate,
                   'batch_size': args.batch_size, 'model': args.model}
    dirname = SAVE_PATH
    dirname, name  = os.path.split(dirname)
    dirname, hypar = os.path.split(dirname)
    dirname, job_type = os.path.split(dirname)
    dirname, model = os.path.split(dirname)
    dirname, project = os.path.split(dirname)
    _ = logsys.create_recorder(hparam_dict=hparam_dict, metric_dict={'best_loss': None},
                               args=args, project=project,
                               entity="szztn951357",
                               group=model,
                               job_type=job_type,
                               name=name,
                               wandb_id=None
                               )

    if save_config and args.local_rank == 0:
        for key, val in vars(args).items():
            logsys.info(f"{key:30s} ---> {val}")
        config_path = os.path.join(logsys.ckpt_root, 'config.json')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                json.dump(vars(args), f)

    return logsys
 