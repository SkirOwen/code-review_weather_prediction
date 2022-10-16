import numpy as np

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from typing import Tuple
from tqdm import tqdm

from weather import BasicBlock, Encoder, Decoder
from weather.weather_object.weather_data import WeatherData

from weather.precip_ai.layers import BasicBlock, Encoder, Decoder


class MHONet(nn.Module):
	""""""
	def __init__(self, bb_in, bb_out, bn, encoder_channels, decoder_channels, out_channel, keep_dim=True,
	             out_size=(479, 1440), padding: int = 0):
		super(MHONet, self).__init__()

		self.bb_in = bb_in
		self.bb_out = bb_out
		self.bn = bn
		self.out_channel = out_channel

		self.encoder_channels = encoder_channels
		self.decoder_channels = decoder_channels

		self.in_basic_block = BasicBlock(*self.bb_in, padding=padding)
		self.out_basic_block = BasicBlock(*self.bb_out, padding=padding)
		self.bottleneck_basic_block = BasicBlock(*self.bn, padding=padding)

		self.encoder = Encoder(self.encoder_channels, padding=padding)
		self.decoder = Decoder(self.decoder_channels, padding=padding)
		self.out_size = out_size

		self.keep_dim = keep_dim

		self.outc = nn.Conv2d(*self.out_channel, kernel_size=1, padding_mode="replicate")

	def forward(self, x):
		x = self.in_basic_block(x)
		skips = self.encoder(x)

		bottle_neck = self.bottleneck_basic_block(skips[-1])
		# print(bottle_neck.shape)

		out = self.decoder(bottle_neck, skips[::-1][1:])

		bb_out = self.out_basic_block(out)
		out = self.outc(bb_out)

		if self.keep_dim:
			out = F.interpolate(out, size=self.out_size)
		return out



	#
	# plt.figure(figsize=(12, 8))
	#
	# plt.subplot(2, 1, 1)
	# plt.imshow(a_cmorph)
	#
	# # plt.subplot(2, 1, 2)
	# # plot(out, "phi(x) : convolutions", True)
	#
	# plt.show()
	#
	# print(out.shape)


def test():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device("cpu")

	date = "20210109"
	cycle = "00"

	wd = WeatherData(date, cycle)
	wd.load_data()
	a_cmorph, a_ecmwf, lat, lon = wd.to_numpy()

	model = MHONet(
		bb_in=(6, 16),
		encoder_channels=(16, 32, 64, 128, 256),
		bn=(256, 256),
		decoder_channels=(256, 128, 64, 32, 16),
		bb_out=(16, 6),
		out_channel=(6, 1),
	)

	# model = Encoder((8, 16, 64))
	model.to(device)
	print("#params", sum(x.numel() for x in model.parameters()))

	# x = torch.randn(1, 6, 721, 1440, device=device)
	x = torch.from_numpy(a_ecmwf[None, :, 121:600, :].astype(np.float32))

	out = model(x)

	import matplotlib.pyplot as plt
	output = out[0]
	output = output.detach().cpu().numpy()
	output = output.transpose((1, 2, 0))

	plt.figure(figsize=(20, 10))
	plt.imshow(output[:, :, 0])

	plt.show()


if __name__ == '__main__':
	test()


