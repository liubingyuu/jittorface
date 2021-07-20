import jittor as jt
from jittor import nn, Module, init
from jittor.nn import Sequential
from collections import namedtuple
import math


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


def xavier_uniform(shape, dtype, gain=1.0):
    assert len(shape)>1

    matsize=1
    for i in shape[2:]:
        matsize *= i
    fan = (shape[1] * matsize) + (shape[0] * matsize)
    bound = gain * math.sqrt(6.0/fan)
    return init.uniform(shape, dtype, -bound, bound)


def xavier_uniform_(var, gain=1.0):
    var.assign(xavier_uniform(tuple(var.shape), var.dtype, gain))


class Flatten(Module):
    def execute(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = jt.norm(input, 2, axis, True)
    output = jt.divide(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv(
            channels, channels // reduction, 1, padding=0, bias=False)

        xavier_uniform_(self.fc1.weight)

        self.relu = nn.ReLU()
        self.fc2 = nn.Conv(
            channels // reduction, channels, 1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.Pool(1, stride=stride, op='maximum')
        else:
            self.shortcut_layer = Sequential(
                nn.Conv(in_channel, depth, (1, 1), stride=stride, bias=False),
                nn.BatchNorm(depth))
        self.res_layer = Sequential(
            nn.BatchNorm(in_channel),
            nn.Conv(in_channel, depth, (3, 3), stride=(1, 1), padding=1, bias=False),
            nn.PReLU(num_parameters=depth),
            nn.Conv(depth, depth, (3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm(depth))

    def execute(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.Pool(1, stride=stride, op='maximum')
        else:
            self.shortcut_layer = Sequential(
                nn.Conv(in_channel, depth, (1, 1), stride=stride, bias=False),
                nn.BatchNorm(depth))
        self.res_layer = Sequential(
            nn.BatchNorm(in_channel),
            nn.Conv(in_channel, depth, (3, 3), stride=(1, 1), padding=1, bias=False),
            nn.PReLU(num_parameters=depth),
            nn.Conv(depth, depth, (3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm(depth),
            SEModule(depth, 16)
        )

    def execute(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            nn.Conv(3, 64, (3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm(64),
            nn.PReLU(num_parameters=64))
        if input_size[0] == 112:
            self.output_layer = Sequential(
                nn.BatchNorm(512),
                nn.Dropout(),
                Flatten(),
                nn.Linear(512 * 7 * 7, 512),
                nn.BatchNorm(512))
        else:
            self.output_layer = Sequential(
                nn.BatchNorm(512),
                nn.Dropout(),
                Flatten(),
                nn.Linear(512 * 14 * 14, 512),
                nn.BatchNorm(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        self._initialize_weights()

    def execute(self, x):
        x = x - 127.5
        x = x * 0.078125
        x = self.input_layer(x)
        x = self.body(x)
        #print("x",x.shape)
        x = self.output_layer(x)
        #print("emb", x.shape)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def IR_50(input_size):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def IR_101(input_size):
    """Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model
