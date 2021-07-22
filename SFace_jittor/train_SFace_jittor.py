import os, argparse

from tensorboardX import SummaryWriter

import jittor as jt
import jittor.optim as optim
from jittor import nn, init

from config import get_config
from image_iter import FaceDataset
from backbone.model_irse import IR_50, IR_101
from backbone.model_mobilefacenet import MobileFaceNet
from head.metrics import SFaceLoss

from util.utils import separate_irse_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy
import math
import time


def xavier_gauss_(var, gain=1.0, mode='avg'):
    shape = tuple(var.shape)
    assert len(shape) > 1

    matsize = 1
    for i in shape[2:]:
        matsize *= i
    if mode == 'avg':
        fan = (shape[1] * matsize) + (shape[0] * matsize)
    elif mode == 'in':
        fan = shape[1] * matsize
    elif mode == 'out':
        fan = shape[0] * matsize
    else:
        raise Exception('wrong mode')
    std = gain * math.sqrt(2.0 / fan)
    return init.gauss_(var, 0, std)


def weight_init(m):
    #print(m)
    if isinstance(m, nn.BatchNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias, 0)
        if hasattr(m, 'running_mean') and m.running_mean is not None:
            init.constant_(m.running_mean, 0)
        if hasattr(m, 'running_var') and m.running_var is not None:
            init.constant_(m.running_var, 1)
    elif isinstance(m, nn.PReLU):
        init.constant_(m.weight, 1)
    else:
        if hasattr(m, 'weight') and m.weight is not None:
            xavier_gauss_(m.weight, gain=2, mode='out')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias, 0)


def schedule_lr(optimizer):
    optimizer.lr /= 10.

    if jt.rank == 0:
        print(optimizer)


def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002:
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('--workers_id', help="gpu ids or cpu", default='cpu', type=str)
    parser.add_argument('--epochs', help="training epochs", default=125, type=int)
    parser.add_argument('--stages', help="training stages", default='35,65,95', type=str)
    parser.add_argument('--lr',help='learning rate',default=1e-1, type=float)
    parser.add_argument('--batch_size', help="batch_size", default=256, type=int)
    parser.add_argument('--data_mode', help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='casia', type=str)
    parser.add_argument('--net', help="which network, ['IR_50', 'IR_101', 'MobileFaceNet']",default='IR_50', type=str)
    parser.add_argument('--head', help="head type, ['SFaceLoss']", default='SFaceLoss', type=str)
    parser.add_argument('--target', help="verification targets", default='lfw,calfw,cplfw,cfp_fp,agedb_30', type=str)
    parser.add_argument('--resume_backbone', help="resume backbone model", default='', type=str)
    parser.add_argument('--resume_head', help="resume head model", default='', type=str)
    parser.add_argument('--outdir', help="output dir", default='test_dir', type=str)
    parser.add_argument('--param_s', default=64.0, type=float)
    parser.add_argument('--param_k', default=80.0, type=float)
    parser.add_argument('--param_a', default=0.8, type=float)
    parser.add_argument('--param_b', default=1.23, type=float)
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    if jt.rank == 0 and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    cfg = get_config(args)

    SEED = cfg['SEED'] # random seed for reproduce results

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train data are stored
    EVAL_PATH = cfg['EVAL_PATH'] # the parent root where your val data are stored
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['IR_50', 'IR_101']
    HEAD_NAME = cfg['HEAD_NAME']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    gpu_nums = len(GPU_ID)
    TARGET = cfg['TARGET']

    if jt.rank == 0:
        print('GPU_ID', GPU_ID)
        print("=" * 60)
        print("Overall Configurations:")
        print(cfg)
        with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
            f.write(str(cfg))
        print("=" * 60)

        writer = SummaryWriter(WORK_PATH)  # writer for buffering intermedium results

    if GPU_ID:
        jt.flags.use_cuda = 1
    else:
        jt.flags.use_cuda = 0
    jt.set_seed(SEED)

    with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
        NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]

    trainloader = FaceDataset(DATA_ROOT, rand_mirror=True, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=gpu_nums)

    vers = get_val_data(EVAL_PATH, TARGET, BATCH_SIZE)
    highest_acc = [0.0 for t in TARGET]


    #======= model & loss & optimizer =======#
    BACKBONE_DICT = {'IR_50': IR_50(INPUT_SIZE),
                     'IR_101': IR_101(INPUT_SIZE),
                     'MobileFaceNet': MobileFaceNet(EMBEDDING_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]

    HEAD = SFaceLoss(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID,
                     s=args.param_s, k=args.param_k, a=args.param_a, b=args.param_b)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_mobilefacenet_bn_paras(BACKBONE) # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_mobilefacenet_bn_paras(HEAD)
    OPTIMIZER = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': WEIGHT_DECAY},
                           {'params': backbone_paras_only_bn}],
                          lr=LR, momentum=MOMENTUM)
    if jt.rank == 0:
        print("Number of Training Classes: {}".format(NUM_CLASS))

        print("=" * 60)
        print(BACKBONE)
        print("{} Backbone Generated".format(BACKBONE_NAME))
        print("=" * 60)
        print("=" * 60)
        print(HEAD)
        print("=" * 60)
        print(OPTIMIZER)
        print("Optimizer Generated")
        print("=" * 60)

        intra_losses = AverageMeter()
        inter_losses = AverageMeter()
        Wyi_mean = AverageMeter()
        Wj_mean = AverageMeter()
        top1 = AverageMeter()

    BACKBONE.apply(weight_init)
    HEAD.apply(weight_init)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        print("=" * 60)
        print(BACKBONE_RESUME_ROOT,HEAD_RESUME_ROOT)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            BACKBONE.load_state_dict(jt.load(BACKBONE_RESUME_ROOT))
            print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            HEAD.load_state_dict(jt.load(HEAD_RESUME_ROOT))
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        print("=" * 60)

    #======= train & validation & save checkpoint =======#
    DISP_FREQ = 20 # frequency to display training loss & acc
    VER_FREQ = 2000
    batch = 0  # batch index

    BACKBONE.train()  # set to training mode
    HEAD.train()
    for epoch in range(NUM_EPOCH):

        if epoch in STAGES:
            schedule_lr(OPTIMIZER)

        if jt.rank == 0:
            last_time = time.time()

        for inputs, labels in iter(trainloader):
            labels = labels.long()
            features = BACKBONE(inputs)

            outputs, loss, intra_loss, inter_loss, WyiX, WjX = HEAD(features, labels)

            OPTIMIZER.zero_grad()
            OPTIMIZER.step(loss)

            prec1 = train_accuracy(outputs.detach(), labels, topk=(1,))
            if jt.in_mpi:
                intra_loss = intra_loss.mpi_all_reduce('mean')
                inter_loss = inter_loss.mpi_all_reduce('mean')
                WyiX = WyiX.mpi_all_reduce('mean')
                WjX = WjX.mpi_all_reduce('mean')
                prec1 = prec1.mpi_all_reduce('mean')
            intra_loss_item = intra_loss.data.item()
            inter_loss_item = inter_loss.data.item()
            WyiX_item = WyiX.data.item()
            WjX_item = WjX.data.item()
            prec1_item = prec1.data.item()
            #embed()
            if jt.rank == 0:
                intra_losses.update(intra_loss_item, inputs.size(0) * gpu_nums)
                inter_losses.update(inter_loss_item, inputs.size(0) * gpu_nums)
                Wyi_mean.update(WyiX_item, inputs.size(0) * gpu_nums)
                Wj_mean.update(WjX_item, inputs.size(0) * gpu_nums)
                top1.update(prec1_item, inputs.size(0) * gpu_nums)

            if ((batch + 1) % DISP_FREQ == 0) and batch != 0 and jt.rank == 0:
                intra_epoch_loss = intra_losses.avg
                inter_epoch_loss = inter_losses.avg
                Wyi_record = Wyi_mean.avg
                Wj_record = Wj_mean.avg
                epoch_acc = top1.avg
                writer.add_scalar("intra_Loss", intra_epoch_loss, batch + 1)
                writer.add_scalar("inter_Loss", inter_epoch_loss, batch + 1)
                writer.add_scalar("Wyi", Wyi_record, batch + 1)
                writer.add_scalar("Wj", Wj_record, batch + 1)
                writer.add_scalar("Accuracy", epoch_acc, batch + 1)

                batch_time = time.time() - last_time
                last_time = time.time()

                print('Epoch {} Batch {}\t'
                      'Speed: {speed:.2f} samples/s\t'
                      'intra_Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                      'inter_Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                      'Wyi {Wyi.val:.4f} ({Wyi.avg:.4f})\t'
                      'Wj {Wj.val:.4f} ({Wj.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, batch + 1, speed=inputs.size(0) * gpu_nums * DISP_FREQ / float(batch_time),
                    loss1=intra_losses, loss2=inter_losses, Wyi=Wyi_mean, Wj=Wj_mean, top1=top1))
                # print("=" * 60)
                intra_losses = AverageMeter()
                inter_losses = AverageMeter()
                Wyi_mean = AverageMeter()
                Wj_mean = AverageMeter()
                top1 = AverageMeter()

            if ((batch + 1) % VER_FREQ == 0) and batch != 0:  # perform validation & save checkpoints (buffer for visualization)
                if jt.rank == 0:
                    lr = OPTIMIZER.lr
                    print("Learning rate %f" % lr)
                    print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                acc = []
                for ver in vers:
                    name, data_set, issame = ver
                    accuracy, std, xnorm, best_threshold, roc_curve = perform_val(EMBEDDING_SIZE, BATCH_SIZE,
                                                                                  BACKBONE, data_set, issame)
                    if jt.rank == 0:
                        buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                        print('[%s][%d]XNorm: %1.5f' % (name, batch + 1, xnorm))
                        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch + 1, accuracy, std))
                        print('[%s][%d]Best-Threshold: %1.5f' % (name, batch + 1, best_threshold))
                    acc.append(accuracy)

                # save checkpoints per epoch
                jt.sync_all()
                if jt.rank == 0 and need_save(acc, highest_acc):
                    BACKBONE.save(os.path.join(WORK_PATH,
                                               "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pkl".format(
                                                   BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                    HEAD.save(os.path.join(WORK_PATH,
                                           "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pkl".format(
                                               HEAD_NAME, epoch + 1, batch + 1, get_time())))
                BACKBONE.train()  # set to training mode

            batch += 1  # batch index
