'''
    Decoder module from MobileNetV2 setting 
'''

from torch import nn, Tensor
from collections import deque
from itertools import islice


def sliding_window_iter(iterable, size):
    '''
        Iterate through iterable using a sliding window of several elements.
        Important: It is a generator!.
        
        Creates an iterable where each element is a tuple of `size`
        consecutive elements from `iterable`, advancing by 1 element each
        time. For example:
        >>> list(sliding_window_iter([1, 2, 3, 4], 2))
        [(1, 2), (2, 3), (3, 4)]
        
        source: https://codereview.stackexchange.com/questions/239352/sliding-window-iteration-in-python
    '''
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:  
        # needed because if iterable was already empty before the `for`,
        # then the window would be yielded twice.
        yield tuple(window)


class MobileNetDecoder(nn.Module):
    r'''
        Decoder module from MobileNetV2 setting.
        This is used for reconstruction of the image from the latent space.
    '''
    def __init__(self, inverted_residual_setting):
        super(MobileNetDecoder, self).__init__()

        decode_path = []
        for setting in sliding_window_iter(inverted_residual_setting[::-1], 2):
            in_setting, out_setting = setting
            _, in_c, _, _ = in_setting
            _, out_c, _, out_s = out_setting
            
            upsample = True if out_s == 2 else False
            decode_path.append(self.__conv_layer__(in_c, out_c, upsample))

        # Include output layer
        decode_path.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_c, 3, kernel_size=3, stride=1, padding=1),
            )
        )
        self.decode_path = nn.Sequential(*decode_path)

    def forward(self, x):
        return self.decode_path(x)

    def __conv_layer__(self, in_channels, out_channels, upsampling=True):
        if upsampling:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    