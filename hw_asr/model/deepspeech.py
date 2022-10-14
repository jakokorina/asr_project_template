from torch import nn
from torch import Tensor
from typing import Union
from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats,
                 n_class,
                 out_conv_channels: int = 32,
                 rnn_hidden_dim: int = 512,
                 rnn_num_layers: int = 5,
                 p_dropout: float = 0.2,
                 rnn_bidirectional: bool = True,
                 **batch):
        super().__init__(n_feats, n_class, **batch)
        self.activation = nn.ReLU()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_conv_channels, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(num_features=out_conv_channels),
            self.activation,
            nn.Conv2d(in_channels=out_conv_channels, out_channels=out_conv_channels, kernel_size=(21, 11),
                      stride=(2, 1)),
            nn.BatchNorm2d(num_features=out_conv_channels),
            self.activation,

        )

        self.rnn = nn.GRU(
            input_size=12 * out_conv_channels,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=p_dropout,
            bidirectional=rnn_bidirectional,

        )
        self.batch_norm = nn.BatchNorm1d(n_feats)

        rnn_out_features = rnn_hidden_dim * 2 if rnn_bidirectional else rnn_hidden_dim
        self.final_layer = nn.Sequential(
            nn.Linear(in_features=rnn_out_features, out_features=n_class, bias=True),
        )

    def forward(self, spectrogram: Tensor, spectrogram_length: Tensor, **batch) -> Union[Tensor, dict]:
        # Convolutional layers
        outputs = self.conv_layers(spectrogram.unsqueeze(1))
        # RNN layers
        batch_size, _, _, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = outputs.view(batch_size, seq_lengths, -1)
        new_length = self.transform_input_lengths(spectrogram_length)
        total_length = new_length.max()
        outputs = nn.utils.rnn.pack_padded_sequence(outputs, new_length,
                                                    batch_first=True,
                                                    enforce_sorted=False)
        outputs, _ = self.rnn(outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, total_length=total_length, batch_first=True)

        # Final layer
        outputs = self.final_layer(outputs)
        return outputs

    def len_after_conv(self, input_lengths, kernel_size, stride):
        numerator = input_lengths - (kernel_size[1] - 1) - 1
        seq_lengths = numerator.float() / float(stride[1])
        seq_lengths = seq_lengths.int() + 1
        return seq_lengths

    def transform_input_lengths(self, input_lengths):
        res_len = self.len_after_conv(input_lengths=input_lengths, kernel_size=(41, 11), stride=(2, 2))
        res_len = self.len_after_conv(input_lengths=res_len, kernel_size=(21, 11), stride=(2, 1))
        return res_len
