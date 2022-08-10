import argparse
import os
from data_utils.dataloader import RibSegDataset
import data_utils.data_aug as data_aug
import torch.nn as nn
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='SegNet',
                        help='model name [default: CLNet]')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=300, type=int, help='Epoch to run [default: 251]')
    parser.add_argument('--learning_rate', default=0.0005, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--root', type=str, default='./dataset/seg_input_10w', help='dataset')
    parser.add_argument('--npoint', type=int, default=30000, help='Point Number [default: 2048]')
    parser.add_argument('--step_size', type=int, default=20, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
    return parser.parse_args()


def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)


    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)


    root = args.root

    TRAIN_DATASET = RibSegDataset(root=root, npoints=args.npoint, split='train',flag_arpe =False)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True)

    TEST_DATASET = RibSegDataset(root=root, npoints=args.npoint, split='test',flag_arpe =False)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False)
    print("The number of labeled training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module('models.'+args.model)

    cls_num = 2
    classifier = MODEL.SegNet(cls_num=cls_num).cuda()
    criterion = MODEL.SegLoss(cls_num=cls_num).cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
        best_loss = checkpoint['best_loss']
        print('Use pretrain model')
        print('best_loss:', best_loss)
    except:
        print('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        best_loss = -99999

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    for epoch in range(start_epoch, args.epoch):
        '''Adjust learning rate and BN momentum'''
        print("epoch:",epoch)
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        '''learning one epoch'''
        ep_loss = 0
        mean_correct = []
        for i, (pc, label) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):        
            pc = pc.data.numpy()
            pc[:,:3] = data_aug.jitter_point_cloud(pc[:,:3], 0.005, 0.01)
            pc[:,:3] = data_aug.random_scale_point_cloud(pc[:,:3], 0.9, 1.1)
            pc = torch.Tensor(pc)
            pc, label = pc.float().cuda(), label.long().cuda()
            pc = pc.transpose(2, 1)
            if cls_num == 2:
              label[label!=0]=1
            
            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(pc)

            seg_pred_choice = pred.contiguous().view(-1, cls_num)
            pred_choice = seg_pred_choice.data.max(1)[1]

            correct = pred_choice.eq(label.flatten()).cpu().sum()
  
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            loss = criterion(pred, label)
            ep_loss += loss

            loss.backward()
            optimizer.step()

        ep_loss /= len(trainDataLoader)
        train_instance_acc = np.mean(mean_correct)
        print('Train accuracy of seg is: %.5f' % train_instance_acc)
        print('Train loss is: %.5f' % ep_loss)

        with torch.no_grad():
            mean_correct_test = []
            for i, (pc, label) in tqdm(
                    enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                pc, label = pc.float().cuda(), label.long().cuda()
                pc = pc.transpose(2, 1)

                if cls_num == 2:
                  label[label!=0]=1

                classifier = classifier.eval()
                pred = classifier(pc)

                seg_pred_choice = pred.contiguous().view(-1, cls_num)
                pred_choice = seg_pred_choice.data.max(1)[1]
                correct = pred_choice.eq(label.flatten()).cpu().sum()
                mean_correct_test.append(correct.item() / (args.batch_size * args.npoint))

            test_instance_acc = np.mean(mean_correct_test)
            print('Test accuracy of seg is: %.5f' % test_instance_acc)
    

        if test_instance_acc >= best_loss:
            best_loss = test_instance_acc
            savepath = str(checkpoints_dir) + '/best_model.pth'
            print('Saving at %s' % savepath)

            state = {
                'best_loss': best_loss,
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            print('Saving model....')

if __name__ == '__main__':
    args = parse_args()
    main(args)