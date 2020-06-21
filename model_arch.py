import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler

import functools, os
import numpy as np

from train_utils import *

basic = 3
# this is from examining the tpch output
dim_dict = {'Seq Scan': num_rel + max_num_attr * 3 + 3 ,
            'Index Scan': num_index + num_rel + max_num_attr * 3 + 3 + 1,
            'Bitmap Heap Scan': num_rel + max_num_attr * 3 + 3 + 32,
            'Bitmap Index Scan': num_index + 3,
            'Sort': 128 + 5 + 32,
            'Hash': 4 + 32,
            'Hash Join': 11 + 32 * 2, 'Merge Join': 11 + 32 * 2,
            'Aggregate': 7 + 32, 'Nested Loop': 32 * 2 + 3, 'Limit': 32 + 3,
            'Subquery Scan': 32 + 3,
            'Materialize': 32 + 3, 'Gather Merge': 32 + 3, 'Gather': 32 + 3}

# basic input:
# plan_width, plan_rows, plan_buffers (ignored), estimated_ios (ignored), total_cost  3

# Sort: sort key [one-hot 128], sort method [one-hot 2];                             2 + 3 = 5
# Hash: Hash buckets, hash algos [one-hot] (ignored);                                1 + 3 = 4
# Hash Join: Join type [one-hot 5], parent relationship [one-hot 3];                 8 + 3 = 11
# Scan: relation name [one-hot ?]; attr min, med, max; [use one-hot instead]         4 + 3 = 7
# Index Scan: never seen one; (Skip)
# Bitmap Heap Scan: 8 + 48 + 3 = 59
# Aggregate: Strategy [one-hot 3], partial mode, operator (ignored)                  4 + 3 = 7

def squared_diff(output, target):
    return torch.sum((output - target)**2)

###############################################################################
#                        Operator Neural Unit Architecture                    #
###############################################################################
# Neural Unit that covers all operators
class NeuralUnit(nn.Module):
    """Define a Resnet block"""

    def __init__(self, node_type, num_layers=5, hidden_size=128, output_size=32,
                 norm_enabled=False):
        """
        Initialize the InternalUnit
        """
        super(NeuralUnit, self).__init__()
        self.node_type = node_type
        self.dense_block = self.build_block(num_layers, hidden_size, output_size,
                                            input_dim = dim_dict[node_type])

    def build_block(self, num_layers, hidden_size, output_size, input_dim):
        """Construct a block consisting of linear Dense layers.
        Parameters:
            num_layers  (int)
            hidden_size (int)           -- the number of channels in the conv layer.
            output_size (int)           -- size of the output layer
            input_dim   (int)           -- input size, depends on each node_type
            norm_layer                  -- normalization layer
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        assert(num_layers >= 2)
        dense_block = [nn.Linear(input_dim, hidden_size), nn.ReLU()]
        for i in range(num_layers - 2):
            dense_block += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        dense_block += [nn.Linear(hidden_size, output_size), nn.ReLU()]

        for layer in dense_block:
            try:
                nn.init.xavier_uniform_(layer.weight)
            except:
                pass
        return nn.Sequential(*dense_block)

    def forward(self, x):
        """ Forward function """
        out = self.dense_block(x)
        return out

###############################################################################
#                               QPP Net Architecture                          #
###############################################################################

class QPPNet():
    def __init__(self, opt):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
                                             else torch.device('cpu:0')
        self.save_dir = opt.save_dir
        self.test = False
        self.test_time = opt.test_time
        self.batch_size = opt.batch_size

        self.last_total_loss = None
        self.last_pred_err = None
        self.pred_err = None
        self.rq = 0
        self.last_rq = 0

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialize the neural units
        self.units = {}
        self.optimizers, self.schedulers = {}, {}
        self.best = 100000
        for operator in dim_dict:
            self.units[operator] = NeuralUnit(operator).to(self.device)
            if opt.SGD:
                optimizer = torch.optim.SGD(self.units[operator].parameters(),
                                            lr=opt.lr, momentum=0.9)
            else:
                optimizer = torch.optim.Adam(self.units[operator].parameters(),
                                             opt.lr) #opt.lr

            if opt.scheduler:
                sc = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                        gamma=opt.gamma)
                self.schedulers[operator] = sc

            self.optimizers[operator] = optimizer


        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.acc_loss = {operator: [self.dummy] for operator in dim_dict}
        self.curr_losses = {operator: 0 for operator in dim_dict}
        self.total_loss = None

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def forward_oneQ_batch(self, samp_batch):
        '''
        Calcuates the loss for a batch of queries from one query template

        compute a dictionary of losses for each operator

        return output_vec, where 1st col is predicted time
        '''
        #print(samp_batch)
        feat_vec = samp_batch['feat_vec']
        input_vec = torch.from_numpy(feat_vec).to(self.device)
        #print(samp_batch['node_type'], input_vec)
        subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _ = self.forward_oneQ_batch(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output_vec),axis=1)
                # first dim is subbatch_size
            else:
                subplans_time.append(torch.index_select(child_output_vec, 1, torch.zeros(1, dtype=torch.long)))

        #print(samp_batch['node_type'], input_vec.size())
        output_vec = self.units[samp_batch['node_type']](input_vec)
        #print(output_vec.shape)
        pred_time = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long)) # pred_time assumed to be the first col
        ## just to get a 1-dim vec of size batch_size out of a batch_sizex1 matrix
        # pred_time = torch.mean(pred_time, 1)

        cat_res = torch.cat([pred_time] + subplans_time, axis=1)
        #print("cat_res.shape", cat_res.shape)
        pred_time = torch.sum(cat_res, 1)
        #print("pred_time.shape", pred_time.shape)
        if self.test_time:
            print(samp_batch['node_type'], pred_time, samp_batch['total_time'])

        loss = (pred_time -
                torch.from_numpy(samp_batch['total_time']).to(self.device)) ** 2
        #print("loss.shape", loss.shape)
        self.acc_loss[samp_batch['node_type']].append(loss)

        # added to deal with NaN
        try:
            assert(not (torch.isnan(output_vec).any()))
        except:
            print("feat_vec", feat_vec, "input_vec", input_vec)
            if torch.cuda.is_available():
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].module.cpu().state_dict())
            else:
                print(samp_batch['node_type'], "output_vec: ", output_vec,
                      self.units[samp_batch['node_type']].cpu().state_dict())
            exit(-1)
        return output_vec, pred_time

    def forward(self, epoch):
        # # self.input is a list of preprocessed plan_vec_dict
        total_loss = torch.zeros(1).to(self.device)
        total_losses = {operator: [torch.zeros(1).to(self.device)] \
                                            for operator in dim_dict}
        if self.test:
            test_loss = []
            pred_err = []

        for idx, samp_dict in enumerate(self.input):
            # first clear prev computed losses
            del self.acc_loss
            self.acc_loss = {operator: [self.dummy] for operator in dim_dict}

            _, pred_time = self.forward_oneQ_batch(samp_dict)
            if self.test:
                tt = torch.from_numpy(samp_dict['total_time']).to(self.device)
                test_loss.append(torch.abs(tt - pred_time))
                pred_err.append(torch.abs(tt - pred_time)/tt)
                if epoch % 50 == 0:
                    print("####### eval by temp: idx {}, test_loss {}, pred_err {}, "\
                      "rq {} ".format(idx, torch.mean(torch.abs(tt - pred_time)).item(),
                              torch.mean(torch.abs(tt - pred_time)/tt).item(),
                              torch.max(torch.cat([tt/(pred_time+torch.finfo(tt.dtype).eps),
                                                   pred_time/(tt+torch.finfo(tt.dtype).eps)])).item()))

                self.rq = max(torch.max(torch.cat([tt/(pred_time+torch.finfo(tt.dtype).eps),
                                                   pred_time/(tt+torch.finfo(tt.dtype).eps)])).item(), self.rq)

            D_size = 0
            subbatch_loss = torch.zeros(1).to(self.device)
            for operator in self.acc_loss:
                #print(operator, self.acc_loss[operator])
                all_loss = torch.cat(self.acc_loss[operator])
                D_size += all_loss.shape[0]
                #print("all_loss.shape",all_loss.shape)
                subbatch_loss += torch.sum(all_loss)

                total_losses[operator].append(all_loss)

            subbatch_loss = torch.mean(torch.sqrt(subbatch_loss/D_size))
            #print("subbatch_loss.shape",subbatch_loss.shape)
            total_loss += subbatch_loss * samp_dict['subbatch_size']

        if self.test:
            all_test_loss = torch.cat(test_loss)
            #print(test_loss[0].shape, test_loss[1].shape, all_test_loss.shape)
            all_test_loss = torch.mean(all_test_loss)
            self.test_loss = all_test_loss

            all_pred_err = torch.cat(pred_err)
            self.pred_err = torch.mean(all_pred_err)
        else:
            self.curr_losses = {operator: torch.mean(torch.cat(total_losses[operator])).item() for operator in dim_dict}
            self.total_loss = torch.mean(total_loss / self.batch_size)
        #print("self.total_loss.shape", self.total_loss.shape)

    def backward(self):
        self.last_total_loss = self.total_loss.item()
        if self.best > self.total_loss.item():
            self.best = self.total_loss.item()
            self.save_units('best')
        self.total_loss.backward()
        self.total_loss = None

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.test = False
        self.forward(epoch)
        # clear prev grad first
        for operator in self.optimizers:
            self.optimizers[operator].zero_grad()

        self.backward()

        for operator in self.optimizers:
            self.optimizers[operator].step()
            if len(self.schedulers) > 0:
                self.schedulers[operator].step()

        self.input = self.test_dataset
        self.test = True
        self.forward(epoch)
        self.last_test_loss = self.test_loss.item()
        self.last_pred_err = self.pred_err.item()
        self.last_rq = self.rq
        self.test_loss, self.pred_err = None, None
        self.rq = 0

    def evaluate(self):
        self.test = True
        self.forward()

    def get_current_losses(self):
        return self.curr_losses

    def save_units(self, epoch):
        for name, unit in self.units.items():
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)

            if torch.cuda.is_available():
                torch.save(unit.module.cpu().state_dict(), save_path)
                unit.to(self.device)
            else:
                torch.save(unit.cpu().state_dict(), save_path)
