import jittor as jt
from jittor import transform

from .verification import evaluate
from image_iter import ValDataset

from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from PIL import Image
import io
import os, pickle, sklearn
import time


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def load_bin(path, batch_size, image_size=[112,112]):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data_loader = ValDataset(bins, issame_list, batch_size, image_size=image_size)
    return data_loader, issame_list


def get_val_pair(path, name, batch_size):
    ver_path = os.path.join(path,name+".bin")
    assert os.path.exists(ver_path)
    data_set, issame = load_bin(ver_path, batch_size)
    print('ver', name)
    return data_set, issame


def get_val_data(data_path, targets, batch_size):
    assert len(targets) > 0
    vers = []
    for t in targets:
        data_set, issame = get_val_pair(data_path, t, batch_size)
        vers.append([t, data_set, issame])
    return vers


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'Sequential' in str(layer.__class__):
            continue
        if 'BatchNorm' in str(layer.__class__):
            paras_only_bn.extend([*layer.parameters()])
        else:
            paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def separate_mobilefacenet_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'mobilefacenet' in str(layer.__class__) or 'Sequential' in str(layer.__class__):
            continue
        if 'BatchNorm' in str(layer.__class__):
            paras_only_bn.extend([*layer.parameters()])
        else:
            paras_wo_bn.extend([*layer.parameters()])

    return paras_only_bn, paras_wo_bn


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize = 14)
    plt.ylabel("TPR", fontsize = 14)
    plt.title("ROC Curve", fontsize = 14)
    plot = plt.plot(fpr, tpr, linewidth = 2)
    buf = io.BytesIO()
    plt.savefig(buf, format = 'jpeg')
    buf.seek(0)
    plt.close()

    return buf


def perform_val(embedding_size, batch_size, backbone, data_set, issame, nrof_folds=10):
    backbone.eval() # switch to evaluation mode

    embeddings_jt = jt.zeros([len(data_set), embedding_size])
    embeddings_jt_flip = jt.zeros([len(data_set), embedding_size])
    with jt.no_grad():
        for i, (batch, batch_flip) in enumerate(data_set):
            output = backbone(batch)
            output_flip = backbone(batch_flip)
            bs_single = batch.shape[0]
            embeddings_jt[i*batch_size+jt.rank*bs_single:i*batch_size+(jt.rank+1)*bs_single] = output.detach()
            embeddings_jt_flip[i*batch_size+jt.rank*bs_single:i*batch_size+(jt.rank+1)*bs_single] = output_flip.detach()
            embeddings_jt.sync()
            embeddings_jt_flip.sync()

    if jt.in_mpi:
        embeddings_jt = embeddings_jt.mpi_all_reduce('add')
        embeddings_jt_flip = embeddings_jt_flip.mpi_all_reduce('add')
    embeddings_list = [embeddings_jt.data, embeddings_jt_flip.data]

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    if jt.rank == 0:
        print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transform.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, acc, std, xnorm, best_threshold, roc_curve_tensor, batch):
    writer.add_scalar('Accuracy/{}_Accuracy'.format(db_name), acc, batch)
    writer.add_scalar('Std/{}_Std'.format(db_name), std, batch)
    writer.add_scalar('XNorm/{}_XNorm'.format(db_name), xnorm, batch)
    writer.add_scalar('Threshold/{}_Best_Threshold'.format(db_name), best_threshold, batch)
    writer.add_image('ROC/{}_ROC_Curve'.format(db_name), roc_curve_tensor, batch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count


def train_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.equal(target.view(1, -1).expand_as(pred))
    #embed()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.multiply(100.0 / batch_size))

    return res[0]
