"""
Adapted from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from data_utils.dataset import our_dataset # todo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_part_seg_msg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')

    parser.add_argument('--data_path', type=str, default=None, help='path where data for dataloader is located')
    parser.add_argument('--picture_path', type=str, default=None, help='path where picture data is located')
    parser.add_argument('--split_seed', type=str, default=None, help='seed of train, val, testsplit')

    return parser.parse_args()


def main(args):

    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    performance_dir = exp_dir.joinpath('performance/')
    performance_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DEFINE DEVICE FOR TRAINING AND OTHER DATA RELATED STUFF'''
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    dataset = our_dataset(args.data_path)
    trainset, valset, testset = torch.utils.data.random_split(dataset, [6500, 2451, 2451],
                                                              generator=torch.Generator().manual_seed(args.seed))

    np.save("testset_indices.npy", testset.indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=10)

    log_string("The number of training data is: %d" % len(trainset))
    log_string("The number of test data is: %d" % len(valset))


    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))

    model = MODEL.get_model().to(device)
    criterion = MODEL.get_loss()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

        # metrics
        train_mse = checkpoint["train_mse"]
        val_mse = checkpoint["val_mse"]

        log_string('Use pretrain model')

    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        model = model.apply(weights_init)

        # metrics
        train_mse = []
        val_mse = []


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    '''
    START TRAINING
    '''

    for epoch in range(start_epoch, args.epoch):
        mean_loss = []

        log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        '''Adjust learning rate'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if epoch % args.step_size == 0:
            print("lr updated to ", lr)

        '''Adjust BN momentum'''
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01

        if epoch % args.step_size == 0:
            print('BN momentum updated to %f' % momentum)

        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        model = model.train()

        '''learning one epoch'''
        for i, (price, columns) in tqdm(enumerate(trainloader), total=len(trainloader), smoothing=0.9):
            optimizer.zero_grad()
            price = price.float().to(device)
            columns = columns.float().to(device)
            pred = model(columns)
            loss = criterion(pred, price)

            mean_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        train_mse.append(np.round(np.mean(mean_loss), 5))
        log_string('Epoch %d train-rMSE: %f' % (epoch + 1, np.sqrt(train_mse[epoch])))

        '''validation set'''
        with torch.no_grad():
            mean_loss = []

            model = model.eval()

            '''apply current model to validation set'''
            for i, (price, columns) in tqdm(enumerate(valloader), total=len(valloader), smoothing=0.9):
                price = price.float().to(device)
                columns = columns.float().to(device)
                pred = model(columns)
                loss = criterion(pred, price)

                mean_loss.append(loss.item())

        val_mse.append(np.round(np.mean(mean_loss), 5))
        log_string('Epoch %d test-rMSE: %f' % (epoch + 1, np.sqrt(val_mse[epoch])))

        if val_mse[epoch] <= np.min(val_mse):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch+1,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        global_epoch += 1

    # save performance measures
    mse_path = str(performance_dir) + '/mse.npy'
    rmse_path = str(performance_dir) + '/rmse.npy'
    mse = np.array([train_mse, val_mse]).T
    rmse = np.sqrt(mse)
    np.save(mse_path, mse)
    np.save(rmse_path, rmse)


if __name__ == '__main__':
    args = parse_args()
    main(args)
