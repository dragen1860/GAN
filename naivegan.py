import os
import numpy as np

import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import dataloader
from torchvision.utils import  make_grid
import PIL
from PIL import Image
from tensorboardX import SummaryWriter



batchsz = 64  # Batch size
lambda_ = 10  # Gradient penalty lambda hyperparameter


class Generator(nn.Module):
	dim = 64

	def __init__(self):
		super(Generator, self).__init__()

		preprocess = nn.Sequential(
			nn.Linear(128, 4 * 4 * 4 * self.dim),
			nn.LeakyReLU(inplace=True),
		)
		block1 = nn.Sequential(
			nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 5),
			nn.LeakyReLU(inplace=True),
		)
		block2 = nn.Sequential(
			nn.ConvTranspose2d(2 * self.dim, self.dim, 5),
			nn.LeakyReLU(inplace=True),
		)
		deconv_out = nn.ConvTranspose2d(self.dim, 1, 8, stride=2)

		self.block1 = block1
		self.block2 = block2
		self.deconv_out = deconv_out
		self.preprocess = preprocess
		self.sigmoid = nn.Sigmoid()

	def forward(self, input):

		output = self.preprocess(input)
		output = output.view(-1, 4 * self.dim, 4, 4)

		output = self.block1(output)

		output = output[:, :, :7, :7]

		output = self.block2(output)

		output = self.deconv_out(output)
		output = self.sigmoid(output)

		return output.view(-1, 1, 28, 28)


class Discriminator(nn.Module):
	dim = 64

	def __init__(self):
		super(Discriminator, self).__init__()

		self.net = nn.Sequential(
			nn.Conv2d(1, self.dim, 5, stride=2, padding=2),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(self.dim, 2 * self.dim, 5, stride=2, padding=2),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(2 * self.dim, 4 * self.dim, 5, stride=2, padding=2),
			nn.LeakyReLU(inplace=True),
		)
		self.output = nn.Linear(4 * 4 * 4 * self.dim, 1)


	def forward(self, input):

		input = input.view(-1, 1, 28, 28)
		out = self.net(input)
		out = out.view(-1, 4 * 4 * 4 * self.dim)
		out = self.output(out)

		return out.view(-1)





def generate_image(step, netG, tb):
	"""
	save generated images to file.
	:param frame: iteration step
	:param netG: G network
	:param tb: tensorboard
	:return:
	"""
	# we only generate 100 images, here shadow global batchsz variable
	batchsz = 100
	noise = torch.randn(batchsz, 128).cuda()
	noisev = autograd.Variable(noise, volatile=True)
	# [b, 128] => [b, 1, 28, 28]
	samples = netG(noisev)

	# [b, 1, 28, 28] => [3, row*28, col*28]
	# make_grid function accept tensor, not variable.
	samples = make_grid(samples.data, batchsz // 10)
	# convert to unsigned int8, [b, 3, 28, 28] => [b, 28, 28, 3]
	# and convert (0, 1) to (0, 255)
	samples_pil = torch.mul(samples, 255).permute(1, 2, 0).byte()
	im = Image.fromarray(samples_pil.cpu().numpy())
	# since 28x28 is too small, we enlarge it by 2 mulplier.
	h, w = im.size
	im_resz = im.resize((2 * h, 2 * w), resample = PIL.Image.LANCZOS)

	# [h, w, 3] => [h, w, 3]/255 => [3, h, w]
	samples_resz = torch.from_numpy(np.array(im_resz) / 255).permute(2, 0, 1)

	im.save('imgs/%s.png'%step)
	tb.add_image('imgs', samples_resz)



# create data loader
mnist_train = datasets.MNIST('data/', train= True, download= True,
                             transform=transforms.Compose([
				                       transforms.ToTensor(),
				                       transforms.Normalize((0.1307,), (0.3081,))])
                             )
loader_train = dataloader.DataLoader(mnist_train, batchsz, shuffle=True)
mnist_val = datasets.MNIST('data/', train= False, download= True,
                             transform=transforms.Compose([
				                       transforms.ToTensor(),
				                       transforms.Normalize((0.1307,), (0.3081,))])
                           )
loader_val = dataloader.DataLoader(mnist_val, batchsz, shuffle=True)


def inf_train_gen():
	"""
	a simple infinite loop data loader.
	:return:
	"""
	while True:
		for images, targets in loader_train:
			yield images


def calc_gradient_penalty(netD, real_data, fake_data):
	"""
	Calcuate gradient of y = netD(data), and penalize the weights for not 1-lipschitz function
	namely, penalize these theta_D if grad(netD(data), data) not in range(0, 1)
	experimentally we wish the grad(netD(data), data) converge to 1 as close as possible, instead of (0, 1)
	:param netD:
	:param real_data: [batchsz:0, 1, 28, 28]
	:param fake_data: [batchsz:1, 1, 28, 28]
	:return:
	"""
	# this is to cope with difference between batchsz:0 and batchsz:1
	min_batchsz = min(real_data.size(0), fake_data.size(0))
	real_data = real_data[:min_batchsz]
	fake_data = fake_data[:min_batchsz]

	alpha = Variable(torch.rand(*real_data.size()).cuda())

	# get a random interpolates connect real_data and fake_data
	interpolates = alpha * real_data + ((1 - alpha) * fake_data)
	interpolates = autograd.Variable(interpolates.data, requires_grad=True)

	# use interpolates to generate
	disc_interpolates = netD(interpolates)

	# disc_interpolates = f(interpolates), NOT f(theta_D) !!!
	# we will calcuate 2nd order derivate on gradient_penalty, which is supported by pytorch from v0.2.0
	# create_graph must be set True for 2nd derivate.
	# grad function will return a tuple
	gradients = autograd.grad(outputs=disc_interpolates,
	                          inputs=interpolates,
	                          grad_outputs=torch.ones(*disc_interpolates.size()).cuda(),
	                          create_graph=True, only_inputs=True)
	gradients = gradients[0]

	# make the norm of gradient converge to 1 as close as possible
	# loss = (grad.norm - 1)^2
	# or max(0, gradident.norm() - 1 )
	gradient_penalty = lambda_ * torch.pow( gradients.norm(2, dim=1) - 1, 2).mean()

	return gradient_penalty



def main():

	tb = SummaryWriter('runs')

	netG = Generator().cuda()
	netD = Discriminator().cuda()
	print(netG)
	print(netD)

	G_mdl_file = 'ckpt/mnist_G.mdl'
	D_mdl_file = 'ckpt/mnist_D.mdl'
	if os.path.exists(G_mdl_file) and os.path.exists(D_mdl_file):
		netD.load_state_dict(torch.load(D_mdl_file))
		netG.load_state_dict(torch.load(G_mdl_file))
		print('loading ckpt done.')
	else:
		print('training from scratch.')


	optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
	optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

	one = torch.FloatTensor([1]).cuda()
	mone = one * -1

	data_iter = inf_train_gen()

	for step in range(200000):

		# (1) Update D network
		for _ in range(5):
			data = next(data_iter)
			real_data = Variable(data.cuda())

			netD.zero_grad()

			# train with real
			D_real = netD(real_data).mean()
			D_real.backward(mone)

			# train with fake
			fake_data = Variable(torch.randn(batchsz, 128).cuda())
			# detach net G
			fake_data = netG(fake_data).detach()
			D_fake = netD(fake_data).mean()
			D_fake.backward(one)

			# train with gradient penalty
			gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)
			gradient_penalty.backward()

			D_cost = D_fake - D_real + gradient_penalty
			Wasserstein_D = (D_real - D_fake)

			# D_cost.backward()
			optimizerD.step()

		# (2) Update G network
		noise = Variable(torch.randn(batchsz, 128).cuda())
		G_output = netG(noise)
		G_D_output = netD(G_output).mean()
		G_D_cost = - G_D_output

		# although here we calculate the gradients of D network, but we don't update weights.
		netG.zero_grad()
		G_D_cost.backward()
		optimizerG.step()

		if step % 50 == 0:
			tb.add_scalar('mnist/train disc cost', D_cost.data[0])
			tb.add_scalar('mnist/train gen cost', G_D_cost.data[0])
			tb.add_scalar('mnist/wasserstein distance', Wasserstein_D.data[0])
			print(step, 'mnist/wasserstein distance:', Wasserstein_D.data[0])

		# Calculate dev loss and generate samples every 100 iters
		if step % 500 == 0:
			val_disc_costs = []
			for images, _ in loader_val:
				images = Variable(images.cuda(), volatile=True)

				D = netD(images).mean()
				val_disc_costs.append(- D.data[0])

			tb.add_scalar('mnist/val disc cost', np.array(val_disc_costs).mean())

			generate_image(step, netG, tb)
			torch.save(netG.state_dict(), G_mdl_file)
			torch.save(netD.state_dict(), D_mdl_file)



if __name__ == '__main__':
	main()