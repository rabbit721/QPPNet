import time, torch
from model_arch import QPPNet
from train_utils import DataSet
import argparse


parser = argparse.ArgumentParser(description='QPPNet Arg Parser')

# Environment arguments
# required
parser.add_argument('-dir', '--save_dir', type=str, default='./saved_model',
                    help='Dir to save model weights (default: ./saved_model)')

parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')

parser.add_argument('--scheduler', action='store_true')
parser.add_argument('--step_size', type=int, default=1000,
                    help='step_size for StepLR scheduler (default: 1000)')

parser.add_argument('--gamma', type=float, default=0.95,
                    help='gamma in Adam (default: 0.95)')

parser.add_argument('--SGD', action='store_true',
                    help='Use SGD as optimizer with momentum 0.9')



parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size used in training (default: 32)')

parser.add_argument('-s', '--start_epoch', type=int, default=0,
                    help='Epoch to start training with (default: 0)')

parser.add_argument('-t', '--end_epoch', type=int, default=200,
                    help='Epoch to end training (default: 200)')

parser.add_argument('-epoch_freq', '--save_latest_epoch_freq', type=int, default=100)

parser.add_argument('-logf', '--logfile', type=str, default='train_loss.txt')

parser.add_argument('--test', action='store_true')
parser.add_argument('--mean_range_dict', type=str)

parser.add_argument('--num_q', type=int, default=22)
parser.add_argument('--num_sample_per_q', type=int, default=320)

def save_opt(opt, logf):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
    logf.write(message)
    logf.write('\n')

if __name__ == '__main__':
    opt = parser.parse_args()

    data_dir = 'res_by_temp/'
    dataset = DataSet(data_dir, opt)

    print("dataset_size", dataset.datasize)
    torch.set_default_tensor_type(torch.FloatTensor)
    qpp = QPPNet(opt)

    total_iter = 0

    logf = open(opt.logfile, 'w+')
    save_opt(opt, logf)

    for epoch in range(opt.start_epoch, opt.end_epoch):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        samp_dicts = dataset.sample_data()
        total_iter += opt.batch_size

        qpp.set_input(samp_dicts)
        qpp.optimize_parameters()
        logf.write("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
                   + '; total_loss: {}; '.format(qpp.last_total_loss))
        print("epoch: " + str(epoch) + "; iter_num: " + str(total_iter) \
              + '; total_loss: {}; '.format(qpp.last_total_loss))

        #if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
        losses = qpp.get_current_losses()
        loss_str = "losses: "
        for op in losses:
          loss_str += str(op) + " [" + str(losses[op]) + "]; "
        print(loss_str)

        logf.write(loss_str + '\n')

        if (epoch + 1) % opt.save_latest_epoch_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch + 1, total_iter))
            qpp.save_units(epoch + 1)

    logf.close()
