from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic, create_logsys
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import torch.distributed as dist
from utils.tools import multistep_demo, multistep_error_plot
from mltool.dataaccelerate import DataSimfetcher
#torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings('ignore')

class InGPUFetcher:
    def __init__(self,data, stamp, batch_size, seq_len, label_len, pred_len, shuffle, time_step=2):
        self.data       = data
        self.stamp      = stamp 
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.label_len  = label_len
        self.time_step  = time_step
        self.length     = (len(self.data) - self.seq_len - (self.time_step-1)*self.pred_len + 1)
        self.index_list  = np.arange(self.length)
        if shuffle:np.random.shuffle(self.index_list)
        self.index_list = torch.LongTensor(self.index_list)
        self.index =  0
        self.length = self.length//batch_size
    def next(self):
        
        s_begin               = self.index_list[self.index*self.batch_size:(self.index+1)*self.batch_size]
        total_sequence_length = self.seq_len + (self.time_step-1)*self.pred_len
        delta                 = torch.arange(total_sequence_length)
        total_sequence_index  = s_begin[:, None] + delta[None]
        sequence_data         = self.data[total_sequence_index] #(B, seq_len + pred_len, D)
        sequence_stamp        = self.stamp[total_sequence_index] #(B, seq_len + pred_len, 4)
        #print(f"sequence_data {sequence_data.shape} mean {sequence_data.mean():.3e} std {sequence_data.std():.3e}")
        if sequence_data.std()< 1e-5:
            print(total_sequence_index)
            torch.save(total_sequence_index,"test.pt")
            raise
        split_rules           = [self.seq_len] + [self.pred_len]*(self.time_step-1)
        splited_sequence      = torch.split(sequence_data,split_rules,1)
        sequence_stamp        = torch.split(sequence_stamp, split_rules, 1)
        self.index+=1
        return splited_sequence, sequence_stamp
        # delta      = torch.arange(self.seq_len)
        # x_index    = s_begin[:,None] + delta[None]
        # delta      = torch.arange(self.seq_len - self.label_len, self.seq_len + self.pred_len)
        # y_index    = s_begin[:,None] + delta[None]
        # seq_x      = self.data[x_index]
        # seq_x_mark = self.stamp[x_index]
        # seq_y      = self.data[y_index]
        # seq_y_mark = self.stamp[y_index]
        # self.index+=1
        # return seq_x, seq_y, seq_x_mark, seq_y_mark 


def parser_compute_graph(compute_graph_set):
    # X0 X1 X2
    # |  |  |
    # x1 x2 x3
    # |  |
    # y2 y3
    # |
    # z3

    if compute_graph_set is None:
        return None, None
    if compute_graph_set == "":
        return None, None
    compute_graph_set_pool = {
        'fwd3_D': ([[1], [2], [3]], [[0, 1, 1, 0.33, "quantity"],
                                     [0, 2, 2, 0.33, "quantity"],
                                     [0, 3, 3, 0.33, "quantity"]]),
        'fwd3_TA': ([[1, 2, 3], [2], [3]], [[0, 1, 1, 0.25, "quantity"],
                                            [0, 2, 2, 0.25, "quantity"],
                                            [1, 2, 2, 0.25, "alpha"],
                                            [1, 3, 3, 0.25, "alpha"]
                                            ]),
        'fwd3_TAL': ([[1, 2, 3], [2], [3]], [[0, 1, 1, 0.25, "quantity"],
                                             [0, 2, 2, 0.25, "quantity"],
                                             [1, 2, 2, 0.25, "alpha_log"],
                                             [1, 3, 3, 0.25, "alpha_log"]
                                             ]),
        'fwd3_KAR': ([[1, 2, 3], [2, 3], [3]], [[0, 1, 1, 0.5, "quantity"],
                                                [0, 2, 2, 0.5, "quantity"],
                                                [1, 2, 2, 0.5, "quantity"],
                                                [1, 3, 3, 0.5, "quantity"],
                                                [2, 3, 3, 0.5, "quantity"]
                                                ]),
        'fwd1_D': ([[1]],   [[0, 1, 1, 1.0, "quantity"]]),
        'fwd1_TA': ([[1, 2], [2]],   [[0, 1, 1, 1.0, "quantity"], [1, 2, 2, 1.0, "alpha"]]),
        'fwd2_D': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity"], [0, 2, 2, 1.0, "quantity"]]),
        'fwd2_D_Log': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_log"], [0, 2, 2, 1.0, "quantity_log"]]),
        'fwd2_D_Rog': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log"], [0, 2, 2, 1.0, "quantity_real_log"]]),
        'fwd2_D_Rog5': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log5"], [0, 2, 2, 1.0, "quantity_real_log5"]]),
        'fwd2_D_Rog9': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log9"], [0, 2, 2, 1.0, "quantity_real_log9"]]),
        'fwd2_D_Rog3': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log3"], [0, 2, 2, 1.0, "quantity_real_log3"]]),
        'fwd2_D_Rog2': ([[1], [2]],   [[0, 1, 1, 1.0, "quantity_real_log2"], [0, 2, 2, 1.0, "quantity_real_log2"]]),
        'fwd2_P': ([[1, 2], [2]], [[0, 1, 1, 1.0, "quantity"],
                                   [0, 2, 2, 1.0, "quantity"],
                                   [1, 2, 2, 1.0, "quantity"]
                                   ]),
        'fwd2_PR': ([[1, 2], [2]], [[0, 1, 1, 0.5, "quantity"],
                                    [0, 2, 2, 0.5, "quantity"],
                                    [1, 2, 2, 1.0, "quantity"]
                                    ]),
        'fwd2_PRO': ([[1, 2], [2]], [[0, 1, 1, 1, "quantity"],
                                     [0, 2, 2, 1, "quantity"],
                                     [1, 2, 2, 0.5, "quantity"]
                                     ]),
        'fwd4_AC': ([[1, 2, 3, 4],
                     [2],
                     [3],
                     [4]],
                    [[0, 1, 1, 1, "quantity"],
                        [1, 2, 2, 1, "quantity"],
                        [1, 3, 3, 1, "quantity"],
                     [1, 4, 4, 1, "quantity"],
                     ]),
        'fwd4_KC_L': ([[1, 2, 3, 4],
                       [2],
                       [3],
                       [4]],
                      [[0, 3, 3, 1, "quantity"],
                          [1, 2, 2, 0.33, "quantity"],
                          [1, 3, 3, 0.33, "quantity"],
                          [1, 4, 4, 0.33, "quantity"],
                       ]),
        'fwd4_AC': ([[1, 2, 3, 4],
                     [2],
                     [3],
                     [4]],
                    [[0, 1, 1, 1, "quantity"],
                        [1, 2, 2, 1, "quantity"],
                        [1, 3, 3, 1, "quantity"],
                     [1, 4, 4, 1, "quantity"],
                     ]),
        'fwd4_C': ([[1, 2, 3, 4],
                    [2],
                    [3],
                    [4]],
                   [[0, 1, 1, 1, "quantity"],
                       [1, 4, 4, 1, "quantity"],
                    ]),
        'fwd4_ABC': ([[1, 2, 3, 4],
                      [2],
                      [3],
                      [4]],
                     [[0, 1, 1, 1, "quantity"],
                         [0, 1, 2, 1, "quantity"],
                         [0, 1, 3, 1, "quantity"],
                         [1, 2, 2, 1, "quantity"],
                         [1, 3, 3, 1, "quantity"],
                         [1, 4, 4, 1, "quantity"],
                      ]),
        'fwd4_ABC_H': ([[1, 2, 3, 4],
                        [2],
                        [3],
                        [4]],
                       [[0, 1, 1, 1, "quantity"],
                           [0, 1, 2, 1, "quantity"],
                           [0, 1, 3, 1, "quantity"],
                        [1, 2, 2, 2, "quantity"],
                        [1, 3, 3, 2, "quantity"],
                        [1, 4, 4, 2, "quantity"],
                        ]),
        'fwd4_ABC_L': ([[1, 2, 3, 4],
                        [2],
                        [3],
                        [4]],
                       [[0, 1, 1, 0.5, "quantity"],
                           [0, 1, 2, 0.5, "quantity"],
                           [0, 1, 3, 0.5, "quantity"],
                        [1, 2, 2, 1, "quantity"],
                        [1, 3, 3, 1, "quantity"],
                        [1, 4, 4, 1, "quantity"],
                        ]),
        'fwd3_ABC': ([[1, 2, 3],
                      [2],
                      [3]],
                     [[0, 1, 1, 1, "quantity"],
                      [1, 2, 2, 1, "quantity"],
                      [1, 3, 3, 1, "quantity"]
                      ]),
        'fwd3_ABC_Log': ([[1, 2, 3],
                          [2],
                          [3]],
                         [[0, 1, 1, 1, "quantity_log"],
                          [1, 2, 2, 1, "quantity_log"],
                          [1, 3, 3, 1, "quantity_log"]
                          ]),
        'fwd3_DC_Log': ([[1, 3],
                         [2],
                         [3]],
                        [[0, 1, 1, 1, "quantity_log"],
                         [0, 2, 2, 1, "quantity_log"],
                         [1, 3, 3, 1, "quantity_log"]
                         ]),
        'fwd3_D_Log': ([[1], [2], [3]],   [[0, 1, 1, 1.0, "quantity_log"], [0, 2, 2, 1.0, "quantity_log"], [0, 3, 3, 1.0, "quantity_log"]]),
        'fwd2_PA': ([[1, 2], [2]], [[0, 1, 1, 1.0, "quantity"],
                                    [0, 2, 2, 1.0, "quantity"],
                                    [1, 2, 2, 1.0, "alpha"]
                                    ]),
        'fwd2_PAL': ([[1, 2], [2]], [[0, 1, 1, 1.0, "quantity"],
                                     [0, 2, 2, 1.0, "quantity"],
                                     [1, 2, 2, 1.0, "alpha_log"]
                                     ]),
        'fwd3_DlongT5': ([[1, 2, 3],
                          [2],
                          [3]],
                         [[0, 1, 1, 1, "quantity"],
                          [0, 1, 2, 1, "quantity"],
                          [0, 1, 3, 1, "quantity"],
                          ], 5),  # <--- in old better version it is another mean
        'fwd3_longT10': ([[1, 2, 3],
                          [2],
                          [3]],
                         [[0, 1, 1, 1, "quantity"],
                         [0, 1, 2, 1, "quantity"],
                         [0, 1, 3, 1, "quantity"],
                         [0, 2, 2, 1, "quantity"],
                         [0, 3, 3, 1, "quantity"],
                          ], "during_valid_normal"),
        'fwd3_D_go10': ([[1], [2], [3]],
                        [],  # <--- no need, will auto deploy for during_valid_normal mode
                        "during_valid_normal_10"),
        'fwd3_D_go10_deltalog': ([[1], [2], [3]],
                                 [],  # <--- no need, will auto deploy for during_valid_normal mode
                                 "during_valid_deltalog_10"),
        'fwd3_D_go10_per_feature': ([[1], [2], [3]],
                                    [],  # <--- no need, will auto deploy for during_valid_normal mode
                                    "during_valid_per_feature_10"),
        'fwd3_D_go10_per_feature_needbase': ([[1], [2], [3]],
                                             [],  # <--- no need, will auto deploy for during_valid_normal mode
                                             "needbase_during_valid_per_feature_10"),
        'fwd3_D_go10_needbase': ([[1], [2], [3]],
                                 [],  # <--- no need, will auto deploy for during_valid_normal mode
                                 "needbase_during_valid_normal_10"),
        'fwd3_D_go10_vallina': ([[1], [2], [3]],
                                [],  # <--- no need, will auto deploy for during_valid_normal mode
                                "vallina_during_valid_normal_10"),
        'fwd3_D_go10_per_feature_vallina': ([[1], [2], [3]],
                                            [],  # <--- no need, will auto deploy for during_valid_normal mode
                                            "vallina_during_valid_per_feature_10"),
        'fwd3_D_go10_per_sample_vallina': ([[1], [2], [3]],
                                           [],  # <--- no need, will auto deploy for during_valid_normal mode
                                           "vallina_during_valid_per_sample_10"),
        'fwd3_D_go10_per_sample_logoffset': ([[1], [2], [3]],
                                             [],  # <--- no need, will auto deploy for during_valid_normal mode
                                             "logoffset_during_valid_per_sample_10"),
        'fwd3_D_go10_runtime_logoffset': ([[1], [2], [3]],
                                          [],  # <--- no need, will auto deploy for during_valid_normal mode
                                          "logoffset_runtime_10"),

    }

    return compute_graph_set_pool[compute_graph_set]
 
class Exp_Long_Term_Forecast(Exp_Basic):
    model_optim = None
    criterion   = None
    scaler      = None
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        compute_graph  = parser_compute_graph(args.compute_graph_set)
        if len(compute_graph)==2:
            self.args.activate_stamps,self.args.activate_error_coef = compute_graph
            self.args.directly_esitimate_longterm_error=0
        else:
            self.args.activate_stamps,self.args.activate_error_coef,self.args.directly_esitimate_longterm_error = compute_graph
            self.args.err_record = {}
            self.args.c1 = self.args.c2 = self.args.c3 = 1

        self.logsys = create_logsys(args) # do not create logsys in Exp_basic, since it is a option of template

    # for an experiment, we need assign the optimizer
    def _select_optimizer(self):
        if self.model_optim is None:
            self.model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return self.model_optim

    # for an experiment, we need assign the criterion
    def _select_criterion(self):
        if self.criterion is None:
            self.criterion = nn.MSELoss()
        return self.criterion

    def _select_scaler(self):
        if self.scaler is None:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        return self.scaler
    
    def once_forward(self, batch_x,batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        # this require the batch_y is from -label_len to pred_len
        if batch_y.shape[1] == self.args.label_len + self.args.pred_len:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
            dec_inp = torch.nn.functional.pad(
                batch_y[:, :self.args.label_len, :], (0, 0, 0, self.args.pred_len))
        else:
            # WARNING: this assume the batch_x and batch_y is from same dataset and overlap on label_len
            #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.nn.functional.pad(batch_x[:, -self.args.label_len:],(0,0,0,self.args.pred_len))
            batch_y_mark = torch.cat([batch_x_mark[:, -self.args.label_len:], batch_y_mark], 1).float()
        assert batch_y_mark.shape[1] == self.args.label_len + self.args.pred_len
        # encoder - decoder
        
        
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        outputs = outputs[0] if self.args.output_attention else outputs
    
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        pred    = outputs
        true    = batch_y
        return pred, true
    
    def valid_one_epoch(self, vali_loader, epoch): # vali one epoch
        logsys = self.logsys
        criterion  = self._select_criterion()
        total_loss = []
        self.model.eval()
        if isinstance(vali_loader,tuple):
            data, stamp = vali_loader
            batch_size  = self.args.valid_batch_size
            prefetcher  = InGPUFetcher(data, stamp, self.args.batch_size, self.args.seq_len, self.args.label_len, self.args.pred_len, False, time_step=self.args.time_step)
            valid_steps = prefetcher.length
        else:
            prefetcher = DataSimfetcher(vali_loader, self.device)
            valid_steps= len(vali_loader)
            batch_size = self.args.valid_batch_size
        inter_b = logsys.create_progress_bar(valid_steps, unit=' stamps', unit_scale=batch_size)
        inter_b.lwrite(f"load everything, start_validating......", end="\r")
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            while inter_b.update_step():
                i = inter_b.now
                sequence_data = prefetcher.next()
                pred, true, loss, abs_loss, iter_info_pool = self.once_iter(sequence_data,'valid')
                total_loss.append(loss)
                if self.args.debug:break
        total_loss = sum(total_loss)/len(total_loss)
        if hasattr(self.model, 'module'):
            for x in [total_loss]:
                dist.barrier()
                dist.reduce(x, 0)
        return total_loss.item()

    def once_iter(self, sequence_data,status):
        criterion   = self._select_criterion()
        if not self.args.activate_stamps:
            assert self.args.time_step == 2 
            (batch_x,batch_y), (batch_x_mark, batch_y_mark) = sequence_data
            # notice we provide batch_y from seq_len to seq_len + pred_len rather seq_len - label_len to seq_len + pred_len
            with torch.cuda.amp.autocast(enabled=self.args.use_amp): # use amp flag auto enable or disable amp
                pred, true = self.once_forward(batch_x,batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
            return pred, true, loss, 0 ,{}
        else:
            sequence_data, sequence_stamp = sequence_data
            # sequence_data  = [t.clone().to(self.device) for t in sequence_data]
            # sequence_stamp = [t.clone().to(self.device) for t in sequence_stamp]
            batch = list(zip(sequence_data,sequence_stamp))
            with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                loss, diff, iter_info_pool = self.run_one_iter_highlevel_fast(batch, status)
            return None, None, loss, diff ,iter_info_pool
        
    def train_one_epoch(self, train_loader, epoch):
        logsys = self.logsys
        self.model.train()
        model_optim = self._select_optimizer()
        scaler      = self._select_scaler()
        accumulation_steps = self.args.accumulation_steps
        if isinstance(train_loader,tuple):
            data, stamp = train_loader
            batch_size  = self.args.batch_size
            prefetcher  = InGPUFetcher(data, stamp, self.args.batch_size, self.args.seq_len, self.args.label_len, self.args.pred_len, True, 
                                       time_step=self.args.time_step)
            train_steps = prefetcher.length

        else:
            batch_size  = train_loader.batch_size
            train_steps = len(train_loader)
            prefetcher  = DataSimfetcher(train_loader,self.device)
        train_loss_one_epoch = []
        data_cost  = []
        train_cost = []
        rest_cost  = []
        
        iter_count = 0
        model_optim.zero_grad()
        time_now = now = time.time()
        inter_b = logsys.create_progress_bar(train_steps, unit=' stamps', unit_scale=batch_size)
        inter_b.lwrite(f"load everything, start_training......", end="\r")
        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
        while inter_b.update_step():
            i = inter_b.now
            iter_count += 1
            sequence_data = prefetcher.next()
            data_cost.append(time.time() - now);now = time.time()
            pred, true, loss, abs_loss, iter_info_pool = self.once_iter(sequence_data,'train')
            train_loss_one_epoch.append(loss.item())
            loss /= accumulation_steps
            scaler.scale(loss).backward()
            

            if (i+1) % accumulation_steps == 0:
                scaler.step(model_optim)
                scaler.update()
                model_optim.zero_grad()

            train_cost.append(time.time() - now);now = time.time()
            if (i + 1) % self.args.trace_freq == 0 or i < 30:
                logpool = {'runtime_loss':loss.item()}
                for key, val in logpool.items():
                    logsys.record(key, val, epoch*train_steps + i, epoch_flag='iter')
                    
                #speed = (time.time() - time_now) / iter_count
                #left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #outstring = f"iters: {i+1:3d}/{train_steps:3d}, epoch: {epoch + 1:3d}/{self.args.train_epochs:3d}| loss: {loss.item():.5f} <= data cost:{np.mean(data_cost[-10:]):.3f} train cost:{np.mean(train_cost[-10:]):.3f} rest cost:{np.mean(rest_cost[-10:]):.3f} speed: {speed:.4f}s/iter; left time: {left_time:.4f}s"
                outstring = f"iters:{i+1: 4d}/{train_steps: 4d} epoch:{epoch + 1: 3d}/{self.args.train_epochs: 3d} | loss: {loss.item(): .5f} <= data cost: {np.mean(data_cost[-10:]): .3f} train cost: {np.mean(train_cost[-10:]): .3f} rest cost: {np.mean(rest_cost[-10:]): .3f}"
                inter_b.lwrite(outstring, end="\r")
                iter_count = 0
                time_now = time.time()
                rest_cost.append(time.time() - now);now = time.time()
        
            if self.args.debug:break
        return np.average(train_loss_one_epoch)

    def train(self, setting = None): # this is full train procedue include train - validation
        # this is train one epoch function
        
        logsys = self.logsys
        self.model.train()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader   = self._get_data(flag='val')
        test_data, test_loader   = self._get_data(flag='test')
        time_now = time.time()
        model_optim = self._select_optimizer()
        early_stop_flag = torch.BoolTensor([False])
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,report_fun=lambda x:logsys.info(x,show=False))
        master_bar = logsys.create_master_bar(self.args.train_epochs)
        metric_dict = logsys.initial_metric_dict(['valid_loss','test_loss'])
        banner = logsys.banner_initial(self.args.train_epochs, logsys.ckpt_root)
        logsys.banner_show(0, logsys.ckpt_root)
        #for epoch in range(self.args.train_epochs):
        best_weight_path = os.path.join(logsys.ckpt_root, 'checkpoint.pth')
        early_stopping2 = EarlyStopping(patience=self.args.patience, verbose=True,report_fun=lambda x:logsys.info(x,show=False))
        data_train  = torch.Tensor(train_data.data_x).to(self.device).float()
        stamp_train = torch.Tensor(train_data.data_stamp).to(self.device).float()
        data_valid  = torch.Tensor(vali_data.data_x).to(self.device).float()
        stamp_valid = torch.Tensor(vali_data.data_stamp).to(self.device).float()
        data_test   = torch.Tensor(test_data.data_x).to(self.device).float()
        stamp_test  = torch.Tensor(test_data.data_stamp).to(self.device).float()
        train_loader= (data_train, stamp_train)
        vali_loader = (data_valid, stamp_valid)
        test_loader = (data_test , stamp_test)
        epoch    = -1
        if self.args.mode == "finetune":
            error_pool_valid = self.fourcast_in_train(steps=self.args.fourcast_step, datasetflag='val',step_length=min(self.args.pred_len, self.args.seq_len),epoch = epoch)                
            for end, error in error_pool_valid.items():self.logsys.record(f"valid_error_{end}", error, epoch, epoch_flag="epoch")
            error_pool_test  = self.fourcast_in_train(steps=self.args.fourcast_step, datasetflag='test',step_length=min(self.args.pred_len, self.args.seq_len),epoch = epoch)
            for end, error in error_pool_test.items():self.logsys.record(f"test_error_{end}", error, epoch, epoch_flag="epoch")
        for epoch in master_bar:
            logsys.record('learning rate',model_optim.param_groups[0]['lr'],epoch, epoch_flag='epoch')
            now = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            train_cost = time.time() - now;now = time.time()
            logsys.info("Epoch: {} train cost time: {}".format(epoch + 1, train_cost), show=False)
            vali_loss  = self.valid_one_epoch(vali_loader,epoch)
            vali_cost = time.time() - now;now = time.time()
            logsys.info("Epoch: {} valid on valid cost time: {}".format(epoch + 1,vali_cost),show=False)
            test_loss  = self.valid_one_epoch(test_loader,epoch)
            vali_cost = time.time() - now;now = time.time()
            logsys.info("Epoch: {} valid on test cost time: {}".format(epoch + 1, vali_cost),show=False)
            logsys.metric_dict.update({'valid_loss': vali_loss, 'test_loss': test_loss}, epoch)
            logsys.banner_show(epoch, logsys.ckpt_root, train_losses=[train_loss])
            logpool = {'train': train_loss,
                       'valid':  vali_loss,
                        'test':  test_loss}
            for key, val in logpool.items():
                logsys.record(key, val, epoch, epoch_flag='epoch')
            
            logsys.info("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(
                epoch + 1, train_loss, vali_loss, test_loss), show=False)

            if self.args.mode == "finetune":
                error_pool_valid = self.fourcast_in_train(steps=self.args.fourcast_step, datasetflag='val',step_length=min(self.args.pred_len, self.args.seq_len),epoch = epoch)
                
                for end, error in error_pool_valid.items():
                    self.logsys.record(f"valid_error_{end}", error, epoch, epoch_flag="epoch")
                if len(error_pool_valid)>0:vali_loss        = error_pool_valid[720]
                error_pool_test  = self.fourcast_in_train(steps=self.args.fourcast_step, datasetflag='test',step_length=min(self.args.pred_len, self.args.seq_len),epoch = epoch)
                for end, error in error_pool_test.items():
                    self.logsys.record(f"test_error_{end}", error, epoch, epoch_flag="epoch")
                if len(error_pool_test)>0:test_loss        = error_pool_test[720]


            if (not self.args.distributed) or (self.args.local_rank == 0):
                early_stopping(vali_loss, self.model, best_weight_path,epoch)
                early_stop_flag = torch.BoolTensor([early_stopping.early_stop])
                if hasattr(self.model, 'module'):
                    for x in [early_stop_flag]:
                        dist.barrier()
                        dist.broadcast(x,0)
                #torch.save({"model":self.model.state_dict(),"epoch":epoch}, os.path.join(logsys.ckpt_root,"latest_checkpoint.pt"))
                #if self.args.mode == "finetune":
                #    torch.save({"model": self.model.state_dict(), "epoch": epoch}, os.path.join(logsys.ckpt_root, "epoch_1.checkpoint.pt"))
            
            if self.args.mode == "finetune" and (not self.args.distributed) or (self.args.local_rank == 0):
                early_stopping2(test_loss, self.model,best_weight_path+'.test', epoch)
                early_stop_flag = torch.BoolTensor([early_stopping2.early_stop])
                if hasattr(self.model, 'module'):
                    for x in [early_stop_flag]:
                        dist.barrier()
                        dist.broadcast(x, 0)
                
            
            if early_stop_flag and self.args.do_early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def test(self, setting, checkpoint=None):
        logsys = self.logsys
        if checkpoint is None:
            logsys.info('loading model')
            path = logsys.ckpt_root
            
            best_weight_path = os.path.join(logsys.ckpt_root, 'checkpoint.pth')
            logsys.info(f'loading model from {best_weight_path}.......')
            state_dict = torch.load(best_weight_path, map_location='cpu')
            if "model" in state_dict:
                epoch = state_dict["epoch"]
                state_dict=state_dict["model"]
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
        preds = []
        trues = []
        folder_path = os.path.join(logsys.ckpt_root, './test_results')
        if not os.path.exists(folder_path):os.makedirs(folder_path)

        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # use amp flag auto enable or disable amp
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    pred, true = self.once_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                    #loss = criterion(pred, true)

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    gt = torch.cat([batch_x[0, :, -1], true[0, :, -1]], 0).detach().cpu().numpy()
                    pd = torch.cat([batch_x[0, :, -1], pred[0, :, -1]], 0).detach().cpu().numpy()
                    name = f"test_image_{i}"
                    visual(gt, pd, os.path.join(folder_path, f"{name}.png"))

        preds = torch.cat(preds).detach().cpu().numpy()
        trues = torch.cat(trues).detach().cpu().numpy()
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join(logsys.ckpt_root, 'results/')
        if not os.path.exists(folder_path):os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logsys.wandblog({"test_mae": mae, "test_mse": mse,
                        "test_rmse": rmse, "test_mape": mape, "test_mspe": mspe})
        logsys.info('mse:{}, mae:{}'.format(mse, mae))
        with open(os.path.join(folder_path, "result.txt"), 'w') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def multistepprediction(self, steps=4, step_length=96, datasetflag='test',checkpoint=None,exceed_stratagy='minimum-rank',epoch=-2,weight_name=None,monitor_alpha=False):
        """
            it should have below functions:
            - do autoregressive prediction and create ground truth file and predection file
                - in fact, the ground truth file is unnecessary.
                - the size of prediction file depend on two variable
                    - block_stride
                    - prediction_length
            the prediction file can be very large, if we compute all the prediction possiblity by block_stride = 1
            thus, we default realized this by using block_stride = prediction_length.
            We dont allow any overlap between prediction sequence and known sequence. 
            If the model provide any overlap, it should be pruned.
            |-------------seq_len-----------|---pred_len---|
            |----input_len----|--label_len--|
                              |--------output_len----------|
                                            |-fourcast_out-|
                         |----------next_input---------|---next_pred---|
            |-> offset <-|

            step_length is just offset
            if step_length == pred_len then 

            |-----------seq_len_1-----------|---pred_len_1--|
                            |-----------seq_len_2-----------|---pred_len_2--|
                                            |-----------seq_len_3-----------|---pred_len_3--|
        """
        self.model.eval()
        time_stamp_stride = self.args.pred_len
        logsys = self.logsys
        if epoch == -2:logsys.info(f"""
            we start a autoregressive forecast proceduce with
                model = {self.args.model} given input ({self.args.seq_len}) label ({self.args.label_len}) -> ouput ({self.args.pred_len})
            the time stamp stride = {time_stamp_stride}
        """)
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        label_len = self.args.label_len
        #assert pred_len <= seq_len, print("now we only support pred_len <= seq_len")
        #assert step_length == pred_len, print("now we only support step_length == pred_len")
        # this mean we only take the first step_length sequence as our true output
        if checkpoint is None and epoch==-2:
            path = logsys.ckpt_root
            name = 'checkpoint.pth'
            if weight_name is not None:name = weight_name
            best_weight_path = os.path.join(logsys.ckpt_root, name)
            logsys.info(f'loading model from {best_weight_path}.......')
            state_dict = torch.load(best_weight_path, map_location='cpu')
            if "model" in state_dict:
                epoch = state_dict["epoch"]
                state_dict=state_dict["model"]
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

        test_data, test_loader = self._get_data(flag=datasetflag)

        ### to fully speed up data loading, we load all data input GPU 
        ##### now in single GPU mode

        data_x       = torch.Tensor(test_data.data_x).to(self.device)
        data_y       = data_x #torch.Tensor(test_data.data_y).to(self.device)
        data_stamp   = torch.Tensor(test_data.data_stamp).to(self.device)


        total_data_len = len(data_stamp) - steps * step_length - seq_len - pred_len 
        if total_data_len < 0:
            logsys.info(
                f"setting steps={steps} step_length={step_length} exceed the test data length {len(data_stamp)}, exit....")
            return None, None
        
        batch_index_list = list(range(0, total_data_len, self.args.valid_batch_size))   +[total_data_len]
        #assert step_length == pred_len, print("the result template requires step_length == pred_len")
        result_length    = seq_len + (steps-1)*step_length + pred_len 
        multistep_result = torch.zeros((total_data_len, result_length, data_x.size(-1))).to(self.device) # (3000, seq_len + pred_len*steps, D)
        multistep_tindex =   np.zeros((total_data_len, result_length)) # (3000, seq_len + pred_len*steps) this declare the time index for this result 
        if  monitor_alpha:
            # (3000, seq_len + pred_len*steps, D)
            multistep_result_rank1 = torch.zeros((total_data_len, result_length, data_x.size(-1))).to(self.device)
        # (3000, seq_len + pred_len*steps) this declare how many times this range is counted
        multistep_counti = torch.zeros((total_data_len, result_length, 1)).to(self.device)
        with torch.no_grad():
            inter_b = logsys.create_progress_bar(len(batch_index_list)-1, unit=' stamps', unit_scale=self.args.valid_batch_size)
            inter_b.lwrite(f"load everything, start_fourcast......", end="\r")
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            while inter_b.update_step():
                i = inter_b.now # start from 1 
                batch_start    = batch_index_list[i-1]# 0 to 32 as one batch; 
                batch_end      = batch_index_list[i] # 32 to 64 as next batch
                
                begin          = torch.arange(batch_start, batch_end) # <--this declare start time_stamp 
                index          = begin[:, None] + torch.arange(0, seq_len + pred_len)[None]
                start          = data_x[index][:, :self.args.seq_len]

                multistep_result[begin, 0:seq_len] = start
                multistep_tindex[begin, 0:seq_len] = begin[:,None] + np.arange(0, self.args.seq_len)[None]
                multistep_counti[begin, 0:seq_len] = 1

                for step_now in range(steps):
                    time_sequence_index = index + step_now*step_length # use this as offset
                    batch_x      = start
                    batch_x_mark = data_stamp[time_sequence_index][:, :self.args.seq_len]
                    batch_y      =     data_y[time_sequence_index][:, -self.args.pred_len-self.args.label_len:]
                    batch_y_mark = data_stamp[time_sequence_index][:, -self.args.pred_len-self.args.label_len:]
                    # |-------------seq_len-----------|---pred_len---|
                    with torch.cuda.amp.autocast(enabled=self.args.use_amp): # use amp flag auto enable or disable amp
                        pred, true = self.once_forward(batch_x,batch_y, batch_x_mark, batch_y_mark)                   
                    if  monitor_alpha: # then we need compute the pred by the T-1 real data
                        with torch.cuda.amp.autocast(enabled=self.args.use_amp): # use amp flag auto enable or disable amp
                            real_batch_x        = data_x[time_sequence_index][:,:self.args.seq_len]
                            rank1_pred, true = self.once_forward(real_batch_x, batch_y, batch_x_mark, batch_y_mark)
                    
                    # pred  = pred[...,:step_length]
                    # true  = true[...,:step_length]
                    # start = torch.cat([batch_x[:, step_length:], pred], 1)
                    
                    # start_index = seq_len + (step_now)*step_length
                    # end_index   = seq_len + (step_now+1)*step_length
                    # multistep_result[begin, start_index:end_index] = pred.detach().cpu().numpy()
                    # multistep_tindex[begin, start_index:end_index] = torch.from_numpy(begin[:,None] + np.arange(start_index, end_index)[None])
                    
                    start_index    = seq_len   + (step_now)*step_length
                    end_index     = start_index + pred_len
                    multistep_tindex[begin, start_index:end_index] = begin[:,None] + np.arange(start_index, end_index)[None]

                    
                            

                    if exceed_stratagy == 'ensamble':
                        multistep_result[begin, start_index:end_index] += pred
                        multistep_counti[begin, start_index:end_index] += 1 
                    elif exceed_stratagy == 'minimum-rank':
                        if step_now ==0:
                            multistep_result[begin, start_index:end_index] = pred
                            if monitor_alpha:multistep_result_rank1[begin, start_index:end_index] = rank1_pred
                        else:
                            multistep_result[begin, end_index - step_length:end_index] = pred[:,- step_length:]
                            if monitor_alpha:
                                multistep_result_rank1[begin, end_index -
                                                       step_length:end_index] = rank1_pred[:, - step_length:]
                        multistep_counti[begin, start_index:end_index] = 1                         
                    else:
                        raise NotImplementedError
                    next_start_bgn_index  = start_index 
                    next_start_end_index  = start_index + seq_len
                    assert not torch.any(multistep_counti[begin, next_start_bgn_index:next_start_end_index]==0)
                    start = multistep_result[begin, next_start_bgn_index:next_start_end_index]/multistep_counti[begin, next_start_bgn_index:next_start_end_index]
                    #start = start.to(self.device)

        assert not torch.any(multistep_counti==0),print(torch.where(multistep_counti==0))
        multistep_result = (multistep_result/multistep_counti).cpu().numpy()
        multistep_tindex = multistep_tindex
        return_list = [multistep_result,multistep_tindex]
        if monitor_alpha:
            return_list.append((multistep_result_rank1/multistep_counti).cpu().numpy())
        return return_list

    def emsemble_prediction(self, steps=4, step_length=96, datasetflag='test',checkpoint=None):
        """
            notice here we do 96 prediction directly, 
            thus for any time_stamp we can create 96 prediction stamps at max
                        ....
            [T-1,T+94]  -> T+127 in [T+95,T+189]
            [T,T+95]  -> T+127 in [T+96,T+190]
            [T+1,T+96]  -> T+127 in [T+97,T+191]
                        ....
            The goal is to record the error for each prediction for any predic len
            we will record three error
            - mean error
            - max error
            - min error
            the recorded result is a tensor shape as 
            (steps*96, 96) which means the statistic error of ith furture prediction from i-j th past.
        """
        raise NotImplementedError

    def plot_multi_step_prediction(self, steps=4, step_length=96, datasetflag='test',
                                   force=False, checkpoint=None,
                                   save_numpy_result=False, folder_path="./debug",weight_name=None,monitor_alpha=False):
        import wandb
        
        logsys = self.logsys
        if not os.path.exists(folder_path):os.makedirs(folder_path)                   
        multistep_result = None
        multistep_result_offline_path = os.path.join(folder_path ,f'{steps}.{step_length}.multistep_result.npy')
        if os.path.exists(multistep_result_offline_path) and not force:
            multistep_result = np.load(multistep_result_offline_path)
        
        
        multistep_tindex=None
        multistep_tindex_offline_path = os.path.join(folder_path ,f'{steps}.{step_length}.multistep_tindex.npy')
        if os.path.exists(multistep_tindex_offline_path) and not force:
            multistep_tindex = np.load(multistep_tindex_offline_path)

        if multistep_result is None:
            multistep_result_list = self.multistepprediction(steps=steps, step_length=step_length, 
                                                                           datasetflag=datasetflag, checkpoint=checkpoint, weight_name=weight_name,monitor_alpha=monitor_alpha)
            if monitor_alpha:
                multistep_result, multistep_tindex , multistep_result_rank1 = multistep_result_list
            else:
                multistep_result , multistep_tindex = multistep_result_list
        if (not os.path.exists(multistep_result_offline_path) or force) and save_numpy_result:
            np.save(multistep_result_offline_path, multistep_result)
        if (not os.path.exists(multistep_tindex_offline_path) or force) and save_numpy_result:
            np.save(multistep_tindex_offline_path, multistep_tindex)
        if monitor_alpha:
            multistep_result_rank1_offline_path = os.path.join(folder_path ,f'{steps}.{step_length}.multistep_result_rank1.npy')
            if (not os.path.exists(multistep_result_rank1_offline_path) or force) and save_numpy_result:
                np.save(multistep_result_rank1_offline_path,
                        multistep_result_rank1)
        
        test_data, test_loader = self._get_data(flag=datasetflag)
        multistep_ground = test_data.data_x[torch.LongTensor(multistep_tindex)]
        
        mean_performance = np.mean((multistep_result - multistep_ground)**2,axis=(0,2))
        for predict_distance, error in enumerate(mean_performance):
            logsys.record("longrange_prediction", error,predict_distance, epoch_flag="future")
        
        figure_list1 = []
        for sample_id, feature_id in [[0, 0], [20, 0]]:
            name = f'{steps}.{step_length}.multistep_result_sample{sample_id:2d}_feature{feature_id}.png'
            line123 = multistep_demo(multistep_result, multistep_ground, self.args.seq_len, steps, step_length,
                                     sample_id=sample_id, feature_id=feature_id,
                                     name=os.path.join(folder_path, name))
            line123 = line123[0] + line123[1] + line123[2]
            logsys.wandblog({name+'_table_pred': wandb.Table(data=line123, columns=["x", "y", "c", "label"])})
            figure_list1.append([line123])

        name = f'{steps}.{step_length}.multistep_error.png'
        error_line = multistep_error_plot(multistep_result, multistep_ground, self.args.seq_len, steps, step_length,
                                          name=os.path.join(folder_path, name))
        logsys.wandblog({name+'_table': wandb.Table(data=error_line, columns=["x", "y"])})
        return figure_list1, error_line

    def plot_multi_step_alpha(self, steps=4, step_length=96, datasetflag='test', checkpoint=None,weight_name=None,monitor_alpha=False):
        import wandb
        monitor_alpha = True
        logsys = self.logsys

        multistep_result_list = self.multistepprediction(steps=steps, step_length=step_length, datasetflag=datasetflag, checkpoint=checkpoint, weight_name=weight_name,monitor_alpha=monitor_alpha)
        multistep_result, multistep_tindex , multistep_result_rank1 = multistep_result_list
        test_data, test_loader = self._get_data(flag=datasetflag)
        multistep_ground = test_data.data_x[torch.LongTensor(multistep_tindex)]
        mean_performance = np.mean((multistep_result - multistep_ground)**2, axis=(0, 2))
        for predict_distance, error in enumerate(mean_performance):
            logsys.record("longrange_prediction", error,
                          predict_distance, epoch_flag="future")

                          
        multistep_ground = multistep_ground.reshape(multistep_ground.shape[0],-1,96,multistep_ground.shape[-1])[:,1:]
        multistep_result = multistep_result.reshape(multistep_ground.shape[0],-1,96,multistep_ground.shape[-1])[:,1:]
        multistep_rank_1 = multistep_result_rank1.reshape(multistep_ground.shape[0],-1,96,multistep_ground.shape[-1])[:,1:]
        assert np.linalg.norm(multistep_result[0][0] - multistep_rank_1[0][0]) == 0
        rank_N_error = np.mean((multistep_ground - multistep_result)**2,axis=(-2,-1))
        #rank_1_error = np.mean((multistep_ground - multistep_rank_1)**2,axis=(-2,-1))
        res_N1_error = np.mean((multistep_result - multistep_rank_1)**2,axis=(-2,-1))

        alpha = np.mean(res_N1_error[:,1:]/rank_N_error[:,:-1],axis=0)
        for predict_distance, a in enumerate(alpha):
            logsys.record(f"alpha_monitor_for_{datasetflag}", a, predict_distance, epoch_flag="step")
        

    def fourcast_in_train(self,steps=4, step_length=96, datasetflag='test', epoch=None):
        assert epoch is not None
        multistep_result , multistep_tindex = self.multistepprediction(steps=steps, step_length=step_length, 
                                              datasetflag=datasetflag, checkpoint=None,epoch = epoch)
        if multistep_result is None:return {}
        test_data, test_loader = self._get_data(flag='test')
        multistep_ground = test_data.data_x[torch.LongTensor(multistep_tindex)]
        mean_performance = np.mean((multistep_result - multistep_ground)**2, axis=(0, 2))
        start = self.args.seq_len
        #end = self.args.seq_len + self.args.pred_len
        fixed_end_range = np.array([96,192,336,720]) + self.args.seq_len
        end_list = set([self.args.seq_len + i *
                        self.args.pred_len for i in range(1,steps)] + list(fixed_end_range))
        error_pool = {}
        for end in end_list:
            if end > len(mean_performance):continue
            error = mean_performance[start:end].mean()
            error_pool[end-start]=error
        return error_pool
    
    def run_one_iter_highlevel_fast(self, batch, status):
        assert self.args.seq_len == self.args.pred_len
        assert batch[0][0].shape[1] == self.args.pred_len
        step_length = self.args.pred_len
        iter_info_pool  = {}
        
        #[[(B,L,7),(B,L,4)],
        # [(B,L,7),(B,L,4)],
        #       .....      ,
        # [(B,L,7),(B,L,4)]]
        # -> (B,T,L,11)  
        # this is only for data Compatible reason. The better way is directly input those data as (B,T,L,11) by slice
        now_level_batch = torch.stack([torch.cat(t,-1) for t in batch],1) # input is a tenosor (B,T,L,D1+D2) where D1 is the feature of series and the D2 is the time stamp feature = 4
        B, L = now_level_batch.shape[:2]
        tshp = now_level_batch.shape[2:]
        # we will do once forward at begin to gain
        # X0 X1 X2 X3
        # |  |  |  |
        # x1 x2 x3 x4
        # |  |  |
        # y2 y3 y4
        # |  |
        # z3 z4
        all_level_batch  = [now_level_batch]
        all_level_record = [list(range(L))]  # [0,1,2,3]]
        ####################################################
    
        fixed_activate_stamps =  self.args.directly_esitimate_longterm_error and 'during_valid' in self.args.directly_esitimate_longterm_error and status == 'valid'
        activate_stamps = [[1, 2, 3], [2], [3]] if fixed_activate_stamps else self.args.activate_stamps
        for step_now in range(len(activate_stamps)):  # generate L , L-1, L-2
            activate_stamp      = activate_stamps[step_now]
            last_activate_stamp = all_level_record[-1]
            picked_stamp        = [last_activate_stamp.index(t-1) for t in activate_stamp]
            target_stamp        = [all_level_record[0].index(t) for t in activate_stamp]
            start               = now_level_batch[:, picked_stamp].flatten(0, 1) # from (B,T,L,D) select the needed stamp for later prediction.
            end                 = all_level_batch[0][:, target_stamp].flatten(0, 1)

            batch_x      = start[..., :-4] # the time information D = 4 #(B, L , 4)
            batch_y      =   end[..., :-4]
            batch_x_mark = start[..., -4:]
            batch_y_mark =   end[..., -4:]

            # use amp flag auto enable or disable amp
            # print(f"step {step_now} x {batch_x.shape} shape {batch_x.shape} mean {batch_x.mean()} std {batch_x.std()}")
            # print(f"step {step_now} y {batch_y.shape} shape {batch_y.shape} mean {batch_y.mean()} std {batch_y.std()}")
            # print(f"step {step_now} m {batch_x_mark.shape} shape {batch_x_mark.shape} mean {batch_x_mark.mean()} std {batch_x_mark.std()}")
            # print(f"step {step_now} n {batch_y_mark.shape} shape {batch_y_mark.shape} mean {batch_y_mark.mean()} std {batch_y_mark.std()}")
            # print("\n"*2)
            pred, true = self.once_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
            # from (B,L,7)  - (B,L,11)
            pred = torch.cat([pred, batch_y_mark[:, -self.args.pred_len:]], -1)
            now_level_batch = pred.reshape(B, len(picked_stamp), *tshp)
            all_level_batch.append(now_level_batch)
            all_level_record.append(activate_stamp)

        ####################################################
        ################ calculate error ###################
        iter_info_pool = {}
        loss = 0
        diff = 0
        loss_count = diff_count = len(self.args.activate_error_coef)

        if not self.args.directly_esitimate_longterm_error:
            for (level_1, level_2, stamp, coef, _type) in self.args.activate_error_coef:
                tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)][...,:-4] # filter out the data, leave the time stamp
                tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)][...,:-4] # filter out the data, leave the time stamp
                if 'quantity' in _type:
                    if _type == 'quantity':
                        error = torch.mean((tensor1-tensor2)**2)
                    elif _type == 'quantity_log':
                        error = ((tensor1-tensor2)**2+1).log().mean()
                    elif _type == 'quantity_real_log':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-2).log().mean()
                    elif _type == 'quantity_real_log5':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-5).log().mean()
                    elif _type == 'quantity_real_log9':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-9).log().mean()
                    elif _type == 'quantity_real_log3':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-3).log().mean()
                        # 1e-2 better than 1e-5.
                        # May depend on the error unit. For 6 hour task, e around 0.03 so we set 0.01 as offset
                        # May depend on the error unit. For 1 hour task, e around 0.006 so we may set 0.01 or 0.001 as offset
                    else:
                        raise NotImplementedError
                elif 'alpha' in _type:
                    last_tensor1 = all_level_batch[level_1-1][:,all_level_record[level_1-1].index(stamp-1)]
                    last_tensor2 = all_level_batch[level_2-1][:,all_level_record[level_2-1].index(stamp-1)]
                    if _type == 'alpha':
                        error = torch.mean(
                            ((tensor1-tensor2)**2) / ((last_tensor1-last_tensor2)**2+1e-4))
                    elif _type == 'alpha_log':
                        error = torch.mean(
                            ((tensor1-tensor2)**2+1).log() - ((last_tensor1-last_tensor2)**2+1).log())
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                iter_info_pool[f"{status}_error_{level_1}_{level_2}_{stamp}"] = error.item()
                loss += coef*error
                if level_1 == 0 and level_2 == stamp:  # to be same as normal train
                    diff += coef*error

        else:
            raise NotImplementedError

        return loss, diff, iter_info_pool

    def run_one_iter_highlevel_batch(self, batch, step_length, status,exceed_stratagy= 'minimum-rank'):
        """
        TODO: 
            when do 1->N task,
            this mode actually do multibranch prediction, for example, if the time expansion N is 5
                 1 -> 6->11
                 2 -> 7->12
            0 -> 3 -> 8->13
                 4 -> 9->14
                 5 ->10->15 
            only the first 5 stamp is generated directly. and then the subsquence become 1 to 1 problem.
        """
        raise NotImplementedError
        # assert model.history_length == 1
        assert self.seq_len == self.pred_len
        assert len(batch) > 1
        assert len(batch) <= len(self.activate_stamps) + 1
        assert self.args.features != "MS"
        iter_info_pool = {}

        # [(B,P,W,H),
        # (B,P,W,H),
        #   ...,
        # (B,P,W,H)]
        #  -> (B,L,P,W,H)
        now_level_batch=batch #now_level_batch = torch.stack(batch, 1)
        # input is a tenosor (B,L,P,W,H)
        # The generated intermediate is recorded as
        # X0 x1 y2 z3
        # X1 x2 y3 z4
        # X2 x3 y4
        # X3 x4
        B, L = now_level_batch.shape[:2]
        tshp = now_level_batch.shape[2:]
        all_level_batch = [now_level_batch]
        all_level_record = [list(range(L))]  # [0,1,2,3]]
        multistep_result = torch.zeros_like(now_level_batch[:,self.seq_len:]) # this will cost some
        multistep_counti = torch.zeros_like(now_level_batch[:,self.seq_len:],dtype=torch.LongTensor)
        ####################################################
        # we will do once forward at begin to gain
        # X0 X1 X2 X3
        # |  |  |  |
        # x1 x2 x3 x4
        # |  |  |
        # y2 y3 y4
        # |  |
        # z3 z4
        # the problem is we may cut some path by feeding an extra option.
        # for example, we may achieve a computing graph as
        # X0 X1 X2 X3
        # |  |  |
        # x1 x2 x3
        # |
        # y2
        # |
        # z3
        # so we need a flag
        ####################################################
        activate_stamps = self.activate_stamps
        fixed_activate_stamps = self.directly_esitimate_longterm_error and 'during_valid' in self.directly_esitimate_longterm_error and status == 'valid'
        activate_stamps = [[1, 2, 3], [2], [3]] if fixed_activate_stamps else self.activate_stamps
        
        
        for step_now in range(len(activate_stamps)):  # generate L , L-1, L-2

            activate_stamp      = activate_stamps[step_now]
            last_activate_stamp = all_level_record[-1]
            picked_stamp        = []
            for t in activate_stamp:picked_stamp.append(last_activate_stamp.index(t-1))


            start = now_level_batch[:, picked_stamp].flatten(0, 1)
            start_index   = (step_now)*step_length
            end_index     = start_index + self.pred_len
            end           = all_level_batch[0][:,start_index:end_index] # <--- the true data is not used in the autoregression, only the time_stamp. Thus can optimize
            #_, _, _, _, start = self.once_forward(i, start)

            batch_x      = start[..., :-4] # 
            batch_x_mark = start[..., -4:] # the time information D = 4 #(B, L , 4)
            batch_y      = end[..., :-4] # 
            batch_y_mark = end[..., -4:] # 


            with torch.cuda.amp.autocast(enabled=self.args.use_amp): # use amp flag auto enable or disable amp
                pred, true = self.once_forward(batch_x,batch_y, batch_x_mark, batch_y_mark)                   
            
            ### only support minimum rank which is also the only right method to create long fut
            assert exceed_stratagy == 'minimum-rank'
            
            pred = torch.cat([pred, batch_y_mark[:,-self.pred_len]],-1) ## from (B,L,7)  - (B,L,11)
            if step_now == 0:
                multistep_result[:,start_index:end_index] = pred
            else:
                multistep_result[:, end_index - step_length:end_index] = pred[:, - step_length:] 
            
            multistep_counti[:, start_index:end_index] = 1
           
            next_start_bgn_index = start_index
            next_start_end_index = start_index + self.seq_len
            assert not torch.any(multistep_counti[:, next_start_bgn_index:next_start_end_index] == 0)
            start = multistep_result[:, next_start_bgn_index:next_start_end_index] / \
                    multistep_counti[:, next_start_bgn_index:next_start_end_index] 
            

            now_level_batch = start.reshape(B, len(picked_stamp), *tshp) 
            all_level_batch.append(now_level_batch) # <-- as you see, we maintain another 
            # all_level_record.append([last_activate_stamp[t]+1 for t in picked_stamp])
            all_level_record.append(activate_stamp)

        ####################################################
        ################ calculate error ###################
        iter_info_pool = {}
        loss = 0
        diff = 0
        loss_count = diff_count = len(self.args.activate_error_coef)

        if not self.directly_esitimate_longterm_error:
            for (level_1, level_2, stamp, coef, _type) in self.args.activate_error_coef:
                tensor1 = all_level_batch[level_1][:,all_level_record[level_1].index(stamp)]
                tensor2 = all_level_batch[level_2][:,all_level_record[level_2].index(stamp)]
                if 'quantity' in _type:
                    if _type == 'quantity':
                        error = torch.mean((tensor1-tensor2)**2)
                    elif _type == 'quantity_log':
                        error = ((tensor1-tensor2)**2+1).log().mean()
                    elif _type == 'quantity_real_log':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-2).log().mean()
                    elif _type == 'quantity_real_log5':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-5).log().mean()

                    elif _type == 'quantity_real_log3':
                        # <---face fatal problem in half precesion due to too small value
                        error = ((tensor1-tensor2)**2+1e-3).log().mean()
                        # 1e-2 better than 1e-5.
                        # May depend on the error unit. For 6 hour task, e around 0.03 so we set 0.01 as offset
                        # May depend on the error unit. For 1 hour task, e around 0.006 so we may set 0.01 or 0.001 as offset
                    else:
                        raise NotImplementedError
                elif 'alpha' in _type:
                    last_tensor1 = all_level_batch[level_1-1][:,all_level_record[level_1-1].index(stamp-1)]
                    last_tensor2 = all_level_batch[level_2-1][:,all_level_record[level_2-1].index(stamp-1)]
                    if _type == 'alpha':
                        error = torch.mean(((tensor1-tensor2)**2) / ((last_tensor1-last_tensor2)**2+1e-4))
                    elif _type == 'alpha_log':
                        error = torch.mean(((tensor1-tensor2)**2+1).log() - ((last_tensor1-last_tensor2)**2+1).log())
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                iter_info_pool[f"{status}_error_{level_1}_{level_2}_{stamp}"] = error.item(
                )
                loss += coef*error
                if level_1 == 0 and level_2 == stamp:  # to be same as normal train
                    diff += coef*error

        else:
            loss, diff, iter_info_pool = lets_calculate_the_coef(
                model, self.directly_esitimate_longterm_error, status, all_level_batch, all_level_record, iter_info_pool)
            # level_1, level_2, stamp, coef, _type
            # a1 = torch.nn.MSELoss()(all_level_batch[3][:,all_level_record[3].index(3)] , all_level_batch[1][:,all_level_record[1].index(3)])\
            #    /torch.nn.MSELoss()(all_level_batch[2][:,all_level_record[2].index(2)] , all_level_batch[0][:,all_level_record[0].index(2)])
            # a0 = torch.nn.MSELoss()(all_level_batch[2][:,all_level_record[2].index(2)] , all_level_batch[1][:,all_level_record[1].index(2)])\
            #    /torch.nn.MSELoss()(all_level_batch[1][:,all_level_record[1].index(1)] , all_level_batch[0][:,all_level_record[0].index(1)])
            # error = esitimate_longterm_error(a0, a1, model.directly_esitimate_longterm_error)
            # iter_info_pool[f"{status}_error_longterm_error_{model.directly_esitimate_longterm_error}"] = error.item()
            # loss += error

        return loss, diff, iter_info_pool, None, None

    