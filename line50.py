import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from torch import optim

from matplotlib import pyplot as plt

class Generator(nn.Module):
	def __init__(self, inc, hc, outc):
		super(Generator, self).__init__()

		self.fc1 = nn.Linear(inc, hc)
		self.fc2 = nn.Linear(hc, hc)
		self.fc3 = nn.Linear(hc, outc)


	def forward(self, x):

		x = F.elu(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		x = self.fc3(x)

		return x



class Discriminator(nn.Module):

	def __init__(self, inc, hc, outc):
		super(Discriminator, self).__init__()

		self.fc1 = nn.Linear(inc, hc)
		self.fc2 = nn.Linear(hc, hc)
		self.fc3 = nn.Linear(hc, outc)

	def forward(self, x):
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.sigmoid(self.fc3(x))

		return x


def decorate_with_diffs(data, exponent = 2.0):
	# [1, 100] => [1, [data, diff]]
	mean = torch.mean(data.data, 1, keepdim=True)
	mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
	diffs = torch.pow(data - Variable(mean_broadcast), exponent)
	return torch.cat([data, diffs], 1)



def test():
	D = Discriminator(200, 50, 1)
	G = Generator(1, 50, 1)
	criteon = nn.BCELoss()

	D_optimizer = optim.Adam(D.parameters(), lr=1e-4)
	G_optimizer = optim.Adam(G.parameters(), lr=1e-4)

	plt.ion()
	plt.figure('real data and generated data')
	plt_real_y = []
	plt_fake_y = []
	plt.plot(plt_fake_y, 'r', label = 'Generated Data')
	plt.plot(plt_real_y, 'g', label = 'Groudtruth Data')
	plt.xlabel('Training Epoch')
	plt.ylabel('Data Distribution')
	plt.legend()

	for epoch in range(20000):

		# train D
		for step in range(2):

			D.zero_grad()

			# [1, 100], with mean = 4, and std = 1.25
			d_real_data = Variable(torch.Tensor(np.random.normal(4, 1.25, (1, 100))))
			# [1, 100] => [1, 200], the normalized data is appended behind original data
			# [1, 200] => [1, 1]
			d_real_decision = D(decorate_with_diffs(d_real_data))
			d_real_error = criteon(d_real_decision, Variable(torch.ones(1, 1)))
			d_real_error.backward()

			# [100, 1], uniformly range from 0 to 1
			d_gen_input = Variable(torch.rand(100, 1))
			# [100, 1] => [100, 1]
			# do not update G network gradient
			d_fake_data = G(d_gen_input).detach()
			# [100, 1] => [1, 100] => [1, 200] => [1, 1]
			d_fake_decision = D(decorate_with_diffs(d_fake_data.t()))
			d_fake_error = criteon(d_fake_decision, Variable(torch.zeros(1, 1)))
			d_fake_error.backward()

			D_optimizer.step()

		# train G
		G.zero_grad()
		# [100, 1]
		gen_input = Variable(torch.rand(100, 1))
		# [100, 1] => [100, 1]
		g_fake_data = G(gen_input)
		# [100, 1] => [1, 100] => [1, 200] => [1, 1]
		# here we have computed the gradients of D network but will not update its gradients since our optimizer
		# consist of G network parameters only
		dg_fake_decision = D(decorate_with_diffs(g_fake_data.t()))
		g_error = criteon(dg_fake_decision, Variable(torch.ones(1, 1)))

		g_error.backward()
		G_optimizer.step()

		if epoch % 30 == 0:
			plt_real_y.append(d_real_data.mean().data[0])
			plt_fake_y.append(d_fake_data.mean().data[0])
			plt.plot(plt_fake_y, 'r')
			plt.plot(plt_real_y, 'g')
			plt.pause(1e-1)










if __name__ == '__main__':
	test()


















