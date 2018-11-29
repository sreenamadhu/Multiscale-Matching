import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import os
from time import strftime, localtime
import numpy as np
import math

def block(kernel = 11):

	net = []
	kernel_size = kernel
	padding = int((kernel_size-1)/2.0)

	net.append(nn.Conv2d(3,16, kernel_size = kernel_size, padding = padding))
	net.append(nn.BatchNorm2d(16))
	net.append(nn.ReLU(True))
	net.append(nn.MaxPool2d(2))

	net.append(nn.Conv2d(16,32, kernel_size = kernel_size, padding = padding))
	net.append(nn.BatchNorm2d(32))
	net.append(nn.ReLU(True))

	return nn.Sequential(*net)



class CrossCNN(nn.Module):
	"""docstring for  CrossCNN"""
	def __init__(self):
		super(CrossCNN, self).__init__()

		self.stream1 = block(7)
		self.stream2 = block(5)
		self.stream3 = block(3)
		self.filter = nn.Sequential(

			nn.Conv2d(32*3, 256, kernel_size = 3, padding = 1),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size = 3,padding = 1),
			nn.ReLU(True),
			nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
			nn.ReLU(True),
			nn.Conv2d(256,1, kernel_size = 3, padding = 1),
			)
		self.fc = nn.Linear(4408, 2)

		self.name = 'CrossCNN_{}filters'.format(64*4)


	def pair_forward(self,x1,x2):

		x1 = self.forward(x1)
		x2 = self.forward(x2)

		difference = (x1-x2)**2
		x = self.filter(difference)

		x = x.view(x.shape[0],-1)

		x = self.fc(x)

		return x

	def forward(self,x):

		x1 = self.stream1(x)
		x1 = x1.div(x1.norm(p=2,dim = 1, keepdim = True))

		x2 = self.stream2(x)
		x2 = x2.div(x2.norm(p=2,dim =1, keepdim =True))
		x3 = self.stream3(x)
		x3 = x3.div(x3.norm(p=2,dim=1, keepdim = True))
		# x4 = self.stream4(x)
		# x4 = x4.div(x4.norm(p=2,dim=1,keepdim = True))
		out = torch.cat((x1,x2,x3),1)
		return out

	def cls_loss(self, x, y):
		loss = F.nll_loss(F.log_softmax(x, dim=1), y)
		return loss

	def loadweight_from(self, pretrain_path):
		pretrained_dict = torch.load(pretrain_path)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict)

	def pair_fit(self, trainloader, validloader, lr=0.001, num_epochs=10,train_log = 'cross_cnn_simple.txt'):
		use_cuda = torch.cuda.is_available()
		if use_cuda:
			self.cuda()
		
		optimizer = optim.Adam(filter(lambda p : p.requires_grad, self.parameters()), lr = lr)
		
		fd = open(train_log, 'w+')

		for epoch in range(num_epochs):

			self.train()
			torch.set_grad_enabled(True)
			train_loss = 0.0
			total = 0
			correct = 0

			for batch_idx, (input1, input2, label) in enumerate(trainloader):

 
				input1 = input1.float()
				input2 = input2.float()
				if use_cuda:
					input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
				optimizer.zero_grad()
				input1, input2, label = Variable(input1), Variable(input2), Variable(label)
				feat = self.pair_forward(input1, input2)
				loss = self.cls_loss(feat, label)
				loss.backward()
				optimizer.step()
				train_loss += loss
				total += label.size(0)
				_, predicted = torch.max(feat.data, 1)
				correct += (predicted == label).sum().item()
				print("    #Iter %3d: Training Loss: %.3f" % (
					batch_idx, loss.data[0]))
				# print(total)
			train_acc = correct/float(total)


			# validate
			self.eval()
			torch.set_grad_enabled(False)
			valid_loss = 0.0
			total = 0
			correct = 0
			tp = 0
			tn = 0
			pos = 0
			neg = 0
			for batch_idx, (input1, input2, label) in enumerate(validloader):
				input1 = input1.float()
				input2 = input2.float()
				if use_cuda:
					input1, input2, label = input1.cuda(), input2.cuda(), label.cuda()
				input1, input2, label = Variable(input1), Variable(input2), Variable(label)
				feat = self.pair_forward(input1,input2)
				valid_loss += self.cls_loss(feat, label)
				# compute the accuracy
				total += label.size(0)

				_, predicted = torch.max(feat.data, 1)
				pos += label.sum()
				neg += (1 - label).sum()
				correct += (predicted == label).sum().item()
				tp += (predicted * label).sum().item()
				tn += ((1 - predicted) * (1 - label)).sum().item()
			valid_acc = correct / float(total)
			tpr = float(tp) / float(pos)
			tnr = float(tn) / float(neg)
			print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
			print("#Epoch {}: Train Loss: {:.4f},Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f}".
				   format(epoch, train_loss / len(trainloader.dataset),train_acc, valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
			fd.write('#Epoch {}: Train Loss: {:.4f},Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}, Valid tpr: {:.4f}, Valid tnr: {:.4f} \n'.
				   format(epoch, train_loss / len(trainloader.dataset),train_acc, valid_loss / len(validloader.dataset), valid_acc, tpr, tnr))
			

			torch.save(self.state_dict(), 'models/pairwise_simple/{}_{:.4f}_epoch-{}.pth'.format(self.name, valid_acc, epoch))


