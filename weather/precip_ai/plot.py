import matplotlib.pyplot as plt


def plot(x, text, normalize=False):
	x = x[0]
	if x.dim() == 4:
		x = x[..., 0]


	nch = x.size(0)

	if normalize:
		x = x - x.view(nch, -1).mean(-1).view(nch, 1, 1)
		x = 0.4 * x / x.view(nch, -1).std(-1).view(nch, 1, 1)

	x = x.detach().cpu().numpy()
	x = x.transpose((1, 2, 0)).clip(0, 1)

	print(x.shape)
	plt.imshow(x)
	plt.axis("off")

	plt.text(0.5, 0.5, text,
	         horizontalalignment='center',
	         verticalalignment='center',
	         transform=plt.gca().transAxes,
	         color='white', fontsize=20)
