from __future__ import print_function
from __future__ import division
import math
import jittor as jt


def xavier_gauss(shape, dtype, gain=1.0, mode='avg'):
    shape = tuple(shape)
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
    return jt.init.gauss(shape, dtype, 0, std)


class SFaceLoss(jt.Module):

    def __init__(self, in_features, out_features, device_id, s=64.0, k=80.0, a=0.80, b=1.23):
        super(SFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.k = k
        self.a = a
        self.b = b
        self.weight = xavier_gauss((out_features, in_features), "float32", gain=2, mode='out')

    def execute(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = jt.nn.matmul_transpose(jt.normalize(input), jt.normalize(self.weight))
        # if self.device_id == None:
        #     cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # else:
        #     x = input
        #     sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
        #     temp_x = x.cuda(self.device_id[0])
        #     weight = sub_weights[0].cuda(self.device_id[0])
        #     cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
        #
        #     for i in range(1, len(self.device_id)):
        #         #print("self.device_id", self.device_id)
        #         temp_x = x.cuda(self.device_id[i])
        #         weight = sub_weights[i].cuda(self.device_id[i])
        #         #print("cosine", i, cosine.size(), cosine)
        #         cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        # --------------------------- s*cos(theta) ---------------------------
        output = cosine * self.s
        # --------------------------- sface loss ---------------------------
        #print("input", cosine.size(), cosine)
        #print("target", label.size(), label)

        one_hot = jt.zeros(cosine.size())
        # if self.device_id != None:
        #     one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), jt.ones((input.shape[0], 1)))

        zero_hot = jt.ones(cosine.size())
        # if self.device_id != None:
        #     zero_hot = zero_hot.cuda(self.device_id[0])
        zero_hot.scatter_(1, label.view(-1, 1), jt.zeros((input.shape[0], 1)))

        #print("one_hot", one_hot)
        #print("zero_hot", zero_hot)

        WyiX = jt.sum(one_hot * output, 1)
        with jt.no_grad():
            # theta_yi = torch.acos(WyiX)
            theta_yi = jt.acos(WyiX / self.s)
            weight_yi = 1.0 / (1.0 + jt.exp(-self.k * (theta_yi - self.a)))
        # theta_yi = jt.acos(WyiX / self.s)
        # weight_yi = 1.0 / (1.0 + jt.exp(-self.k * (theta_yi - self.a)))
        intra_loss = - weight_yi * WyiX

        Wj = zero_hot * output
        with jt.no_grad():
            # theta_j = torch.acos(Wj)
            theta_j = jt.acos(Wj / self.s)
            weight_j = 1.0 / (1.0 + jt.exp(self.k * (theta_j - self.b)))
        # theta_j = jt.acos(Wj / self.s)
        # weight_j = 1.0 / (1.0 + jt.exp(self.k * (theta_j - self.b)))
        inter_loss = jt.sum(weight_j * Wj, 1)

        # print("intra loss", intra_loss.size())
        # print("inter loss", inter_loss.size())
        loss = intra_loss.mean() + inter_loss.mean()
        Wyi_s = WyiX / self.s
        Wj_s = Wj / self.s
        return output, loss, intra_loss.mean(), inter_loss.mean(), Wyi_s.mean(), Wj_s.mean()


