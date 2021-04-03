import torch
from torch import nn

import numpy as np

from .helpers import normalize_timestamps, compute_event_image
from .layers import GConv2d, GConv2d_right, ResNetBlock, UpsampleBlock, QuantizationLayer, InverseGradientLayer

_BASE_CHANNELS = 64


class EV_FlowNet(nn.Module):
    def __init__(self, device, input_size=4, activation=nn.ReLU()):
        super(EV_FlowNet, self).__init__()
        self.device = device
        in_size = 4
        enc_depth = 4
        tr_depth = 2
        tr_res_depth = 2
        kernel_size = (3, 3)
        # encoder
        sizes = [input_size]
        self.encoder_blocks = nn.ModuleList()
        for i in range(enc_depth):
            sizes.append(_BASE_CHANNELS * 2**i)
            self.encoder_blocks.append(GConv2d_right(sizes[-2], sizes[-1], kernel_size, 2))
        # transition
        self.transition_blocks = nn.ModuleList()
        for i in range(tr_depth):
            block = ResNetBlock(sizes[-1], kernel_size=kernel_size, depth=tr_res_depth, activation=activation)
            self.transition_blocks.append(block)

        # decoder
        self.decoder_blocks = nn.ModuleList()
        self.flow_output_blocks = nn.ModuleList()
        sizes[0] = 32
        for i in range(enc_depth):
            in_size = 2 * sizes[-1-i] + (2 if i>0 else 0)
            out_size = sizes[-2-i]
            upsample_block = UpsampleBlock(in_size, out_size, kernel_size=kernel_size, activation=activation)
            self.decoder_blocks.append(upsample_block)
            conv_block = GConv2d(out_size, 2, kernel_size=(1,1), activation=activation, use_bn=False)
            self.flow_output_blocks.append(conv_block)

    def _get_result(self, flow, outsize):
        return tuple(f[..., :s[0], :s[1]] for f, s in zip(flow, outsize))

    def _extend_size(self, imsize):
        return tuple(map(lambda x: ((x - 1) // 16 + 1) * 16, imsize))

    def encode(self, layer_inputs, intermediate_output):
        for enc_block in self.encoder_blocks:
            prev_input = layer_inputs[-1]
            encoded_input = enc_block(prev_input)
            if intermediate_output:
                intermediate_output[f'enc_{len(layer_inputs)-1}'] = encoded_input
            layer_inputs.append(encoded_input)

    def transit(self, encoded_input, intermediate_output):
         # transition
        hidden_state = encoded_input
        for idx, transition_block in enumerate(self.transition_blocks):
            hidden_state = transition_block(hidden_state)
            if intermediate_output:
                intermediate_output[f'tr_{idx}'] = hidden_state
        return hidden_state

    def decode(self, hidden_state, layer_inputs, intermediate_output):
        outputs = []
        layer_inputs = reversed(layer_inputs[1:])
        for idx, (layer_input, decoder_block, flow_out_block) in enumerate(zip(layer_inputs, self.decoder_blocks, self.flow_output_blocks)):

            hidden_state = torch.cat((hidden_state, layer_input), 1)
            if intermediate_output:
                intermediate_output[f'dec_cat_{idx}'] = hidden_state
            hidden_state = decoder_block(hidden_state)
            if intermediate_output:
                intermediate_output[f'dec_op_{idx}'] = hidden_state
            internal_flow = flow_out_block(hidden_state)
            if intermediate_output:
                intermediate_output[f'dec_flow_arth_{idx}'] = internal_flow
                current_out = torch.tanh(internal_flow).clone() # clone is required for backward pass
            else:
                current_out = torch.tanh_(internal_flow)

            current_out = current_out.mul_(256.)
            hidden_state = torch.cat((hidden_state, current_out), 1)
            outputs.append(current_out)
        return outputs


    def compute_outsizes(self, imsize):
        blocks_count = len(self.encoder_blocks)
        denominators = 2 ** np.arange(blocks_count - 1, -1, -1)
        imsizes_x = np.tile(imsize[0], blocks_count) // denominators
        imsizes_y = np.tile(imsize[1], blocks_count) // denominators
        outsizes = zip(imsizes_x, imsizes_y)
        return outsizes

    def forward(self, network_input, imsize, intermediate=False, domain_descriminator=None):
        # compute event_image
        if type(network_input) ==  tuple:
            events, start, stop = network_input
            extended_size = self._extend_size(imsize)
            with torch.no_grad():
                network_input = compute_event_image(events,
                                         start,
                                         stop,
                                         extended_size,
                                         device=self.device,
                                         dtype=torch.float32)

        if intermediate:
            intermediate_output = {'input': network_input}
        else:
            intermediate_output = None

        layer_inputs = [network_input]
        self.encode(layer_inputs, intermediate_output)
        hidden_state = self.transit(layer_inputs[-1], intermediate_output)
        outputs =self.decode(hidden_state, layer_inputs, intermediate_output)

        # shrink image to original size
        outsizes = self.compute_outsizes(imsize)
        results = [self._get_result(outputs, outsizes)]

        if intermediate:
            results.append(intermediate_output)

        if domain_descriminator:
            results.append(domain_descriminator(hidden_state))

        return results[0] if len(results) == 1 else results


class EV_OFlowNet(nn.Module):
    def __init__(self,
                 device,
                 voxel_dimension=9,
                 mlp_layers=[1, 30, 30, 1],
                 quantization_activation=nn.LeakyReLU(negative_slope=0.1),
                 predictor_activation=nn.ReLU()):
        super(EV_OFlowNet, self).__init__()
        self.device = device
        # events representation
        self.quantization_layer = QuantizationLayer(voxel_dimension,
                                                    mlp_layers,
                                                    quantization_activation)
        self.predictor = EV_FlowNet(device=self.device,
                                   input_size=voxel_dimension * 2,
                                   activation=predictor_activation)

    def forward(self,
                events,
                start,
                stop,
                imsize,
                intermediate=False,
                domain_descriminator=False):

        events = normalize_timestamps(events, start, stop)
        batch_size = start.numel()
        xb = self.quantization_layer(events, imsize, batch_size)
        return self.predictor(xb, imsize, intermediate=intermediate, domain_descriminator=domain_descriminator)

    def get_output_sizes(self, imshape):
        return self.predictor.get_output_sizes(imshape)



class ModelWithDomainAdoption(nn.Module):
    def __init__(self, device, use_oflow=True):
        super(ModelWithDomainAdoption, self).__init__()

        if use_oflow:
            self.predictor = EV_OFlowNet(device=device)
        else:
            self.predictor = EV_FlowNet(device=device)

        self.domain_descriminator = nn.Sequential(
            InverseGradientLayer(),
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def descriminate(self, state):
        return self.domain_descriminator(state)

    def forward(self, events, start, stop, imsize, intermediate=False, descriminate_domain=False):
        if descriminate_domain:
            return self.predictor.forward(events, start, stop, imsize, intermediate, self.domain_descriminator)
        else:
            return self.predictor.forward(events, start, stop, imsize, intermediate, False)
