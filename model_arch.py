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
dim_dict = {'Seq Scan': num_rel + max_num_attr + 3 , 'Sort': 128 + 5 + 32, 
            'Hash': 4 + 32,
            'Hash Join': 10 + 32 * 2, 'Merge Join': 10 + 32 * 2,
            'Aggregate': 7 + 32, 'Nested Loop': 32 * 2 + 3, 'Limit': 32 + 3,
            'Subquery Scan': 32 + 3,
            'Materialize': 32 + 3, 'Gather Merge': 32 + 3, 'Gather': 32 + 3}

# basic input:
# plan_width, plan_rows, plan_buffers (ignored), estimated_ios (ignored), total_cost  3

# Sort: sort key [one-hot 128], sort method [one-hot 2];                             2 + 3 = 5
# Hash: Hash buckets, hash algos [one-hot] (ignored);                                1 + 3 = 4
# Hash Join: Join type [one-hot 4], parent relationship [one-hot 3];                 7 + 3 = 10
# Scan: relation name [one-hot ?]; attr min, med, max; [use one-hot instead]         4 + 3 = 7
# Index Scan: never seen one; (Skip)
# Aggregate: Strategy [one-hot 3], partial mode, operator (ignored)                  4 + 3 = 7


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
            nn.init.xavier_uniform(layer.weight)
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
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialize the neural units
        self.units = {}
        self.optimizers, self.schedulers = {}, {}
        self.best = 100000
        for operator in dim_dict:
            self.units[operator] = NeuralUnit(operator).to(self.device)
            optimizer = torch.optim.Adam(self.units[operator].parameters(),
                                         opt.lr) #opt.lr
            #optimizer = torch.optim.SGD(self.units[operator].parameters(),
            #                            lr=opt.lr, momentum=0.9)
            sc = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

            self.optimizers[operator] = optimizer
            self.schedulers[operator] = sc

        self.loss_fn = nn.MSELoss()
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.acc_loss = {operator: [self.dummy] for operator in dim_dict}
        self.curr_losses = {operator: 0 for operator in dim_dict}

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def forward_oneQ_batch(self, samp_batch):
        '''
        Calcuates the loss for a batch of queries from one query template

        compute a dictionary of losses for each operator

        return output_vec, where 1st col is predicted time
        '''
        input_vec = samp_batch['feat_vec']
        input_vec = torch.from_numpy(input_vec).to(self.device)
        #print(samp_batch['node_type'], input_vec)
        subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec = self.forward_oneQ_batch(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = torch.cat((input_vec, child_output_vec),axis=1)
                # first dim is subbatch_size
            else:
                subplans_time.append(torch.index_select(child_output_vec, 1, torch.zeros(1, dtype=torch.long)))

        #print(samp_batch['node_type'], input_vec.size())
        output_vec = self.units[samp_batch['node_type']](input_vec)
        pred_time = torch.index_select(output_vec, 1, torch.zeros(1, dtype=torch.long)) # pred_time assumed to be the first col
        ## just to get a 1-dim vec of size batch_size out of a batch_sizex1 matrix
        # pred_time = torch.mean(pred_time, 1)
        
        cat_res = torch.cat([pred_time] + subplans_time, axis=1)
        #print("cat_res.shape", cat_res.shape)
        pred_time = torch.mean(cat_res, 1)
        #print("pred_time.shape", pred_time.shape)
        #print(output_vec, samp_batch['total_time'])

        loss = self.loss_fn(pred_time,
                            torch.from_numpy(samp_batch['total_time']).to(self.device))
        #print(loss.shape)
        self.acc_loss[samp_batch['node_type']].append(loss.unsqueeze(0))
        return output_vec

    def forward(self):
        # first clear prev computed losses
        del self.acc_loss
        self.acc_loss = {operator: [self.dummy] for operator in dim_dict}

        # # self.input is a list of preprocessed plan_vec_dict
        for samp_dict in self.input:
            _ = self.forward_oneQ_batch(samp_dict)

    def backward(self, batch_size):
        total_loss = torch.zeros(1).to(self.device)
        for operator in self.acc_loss:
            #print(operator, self.acc_loss[operator])
            op_loss = torch.sum(torch.cat(self.acc_loss[operator])) / batch_size
            self.curr_losses[operator] = op_loss.item()
            total_loss += op_loss

        total_loss = torch.sqrt(total_loss)
        print("total loss: ", total_loss.item())
        if self.best > total_loss.item():
            self.best = total_loss.item()
            self.save_units('best')
        total_loss.backward()

    def optimize_parameters(self, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.forward()
        self.backward(batch_size)
        for operator in self.optimizers:
            self.optimizers[operator].step()
            self.schedulers[operator].step()

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

    '''
    def optimize_parameters(self, batch_size):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
    '''
