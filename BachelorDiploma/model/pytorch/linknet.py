from fastai.vision import nn, models, F, Tensor
from functools import partial

__decoder_outputs__ = []
__encoder_outputs__ = []
__pre_encoder_outputs__ = []
__classifier_middle__ = []


def __rescale__(inp_tensor) -> Tensor:
    mn = inp_tensor.min(dim=1).values.min(dim=1).values
    for i in range(mn.size()[0]):
        inp_tensor[i] -= mn[i]

    mx = inp_tensor.max(dim=1).values.max(dim=1).values
    for i in range(mx.size()[0]):
        inp_tensor[i] *= 255
        inp_tensor[i] /= mx[i]

    return inp_tensor


def __output__(outputs_container: list, decoder_number: int, batch_number: int,
               in_batch_position: int = 0, rescale: bool = False) -> Tensor:
    output = outputs_container[batch_number][decoder_number][in_batch_position]

    if rescale:
        __rescale__(output)

    return output


__decoder_output__ = partial(__output__, __decoder_outputs__)

__decoder_high_output_position__ = 0
__decoder_middle_output_position__ = 1
__decoder_first_output_position__ = 3

decoder_high_output = partial(__decoder_output__, __decoder_high_output_position__)
decoder_middle_output = partial(__decoder_output__, __decoder_middle_output_position__)
decoder_first_output = partial(__decoder_output__, __decoder_first_output_position__)

__encoder_output__ = partial(__output__, __encoder_outputs__)

__encoder_first_output_position__ = 0
__encoder_middle_output_position__ = 1
__encoder_low_output_position__ = 3

encoder_first_output = partial(__encoder_output__, __encoder_first_output_position__)
encoder_middle_output = partial(__encoder_output__, __encoder_middle_output_position__)
encoder_low_output = partial(__encoder_output__, __encoder_low_output_position__)

__pre_encoder_output__ = partial(__output__, __pre_encoder_outputs__)
pre_encoder_output = partial(__pre_encoder_output__, 0)

__classifier_middle_output__ = partial(__output__, __classifier_middle__)
classifier_middle_output = partial(__classifier_middle_output__, 0)


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4,
                                          stride=2, padding=1, output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x


def __print_info__(f5, x_out):
    print(
        "Pre-softmax state; Size: {0},\n1. {1},\n2. {2}\n".format(
            f5[0].size(),
            f5[0, :, 5, 128].to('cpu', float),
            f5[0, :, 128, 128].to('cpu', float)))

    only_softmax = F.softmax(f5, dim=1)

    print(
        "Post-softmax state; Size: {0},\n1. {1},\n2. {2}\n".format(
            only_softmax[0].size(),
            only_softmax[0, :, 5, 128].to('cpu', float),
            only_softmax[0, :, 128, 128].to('cpu', float)))

    print(
        "Post-logsoftmax state; Size: {0},\n1. {1},\n2. {2}\n\n\n".format(
            x_out[0].size(),
            x_out[0, :, 5, 128].to(
                'cpu', float),
            x_out[0, :, 128, 128].to('cpu', float)))


def __save_outputs__(outputs_container: list, outputs: list) -> None:
    outputs_container += [outputs]


__save_decoder_outputs__ = partial(__save_outputs__, __decoder_outputs__)
__save_encoder_outputs__ = partial(__save_outputs__, __encoder_outputs__)
__save_pre_encoder_outputs__ = partial(__save_outputs__, __pre_encoder_outputs__)
__save_classifier_outputs__ = partial(__save_outputs__, __classifier_middle__)


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, pretrained=True, debug_info=False,
                 save_decoder_outputs=False, save_encoder_outputs=False):
        super().__init__()
        assert num_channels == 3
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.dropout1 = nn.Dropout2d(0.5, False)
        self.dropout2 = nn.Dropout2d(0.3, False)

        self.with_debug_info = debug_info
        self.save_encoder = save_encoder_outputs
        self.save_decoder = save_decoder_outputs

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        # dr1 = self.dropout1(f2)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # dr2 = self.dropout2(f4)
        f5 = self.finalconv3(f4)

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5

        _ = self.with_debug_info and __print_info__(f5, x_out)
        _ = self.save_encoder and __save_pre_encoder_outputs__([x])
        _ = self.save_encoder and __save_encoder_outputs__([e1, e2, e3, e4])
        _ = self.save_decoder and __save_decoder_outputs__([d1, d2, d3, d4])
        _ = self.save_decoder and __save_classifier_outputs__([dr1])

        return x_out
