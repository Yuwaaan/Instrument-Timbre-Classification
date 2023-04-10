import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os
import sys
import random
from pydub import AudioSegment

os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_gpu = torch.cuda.is_available()
print(use_gpu)
if_gpu = torch.cuda.is_available()  # whether available
print(if_gpu)
gpu_number = torch.cuda.current_device()
print(gpu_number)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


dataset = h5py.File('train_data_whole2.h5', 'r') #

vector_size = dataset.attrs['vector_size']
num_of_labels = len(dataset)
num_of_tracks = sum([dataset[key].shape[0] for key in dataset])

print(num_of_labels)
print(num_of_tracks)
print([dataset[key].shape[0] for key in dataset])

keys = [key for key in dataset] # The keys of the 2 instruments to be used
print(keys)

# Prepare data for training and testing
features = np.zeros((num_of_tracks, vector_size[0], vector_size[1]), dtype=np.float32)
labels = np.zeros((num_of_tracks, len(dataset)), dtype=np.float32)

i = 0
for ki, k in enumerate(dataset):
	features[i:i + len(dataset[k])] = np.nan_to_num(dataset[k])
	labels[i:i + len(dataset[k]), ki] = 1
	i += len(dataset[k])

shuffled_features = np.empty(features.shape, dtype=features.dtype)
shuffled_labels = np.empty(labels.shape, dtype=labels.dtype)
permutation = np.random.permutation(len(features))
for old_index, new_index in enumerate(permutation):
    shuffled_features[new_index] = features[old_index]
    shuffled_labels[new_index] = labels[old_index]
    
features = shuffled_features
labels = shuffled_labels

X_train, X_eval, Y_train, Y_eval = train_test_split(features, labels, test_size=0.1, random_state=1337)

X_train_torch = torch.from_numpy(X_train).to(device)
X_eval_torch = torch.from_numpy(X_eval).to(device)
Y_train_torch = torch.from_numpy(Y_train).to(device)
Y_eval_torch = torch.from_numpy(Y_eval).to(device)

trainset = torch.utils.data.TensorDataset(X_train_torch, Y_train_torch)
evalset = torch.utils.data.TensorDataset(X_eval_torch, Y_eval_torch)

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
eval_dataloader = torch.utils.data.DataLoader(evalset, batch_size=100, shuffle=True, num_workers=0)

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		# 1*130*128
		# (130-5) + 1 = 126
		# (128-5) + 1 = 124
		# 6*126*124

		self.pool = nn.MaxPool2d(2, 2)
		# first max-pooling
        # (126-2)/2 + 1 = 63
		# (124-2)/2 + 1 = 62
		# 6*63*62
        
        # second max-pooling
		# (59-2)/2 + 1 = 29.5(29)
		# (58-2)/2 + 1 = 29
		# 6*29*29 = 13456

		self.conv2 = nn.Conv2d(6, 16, 5)
		# (63-5) + 1 = 59
		# (62-5) + 1 = 58
		# 16*59*58

		self.fc1 = nn.Linear(928, 120) #13456
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, num_of_labels)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 928) #13456
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Net2(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 3)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(3, 3)    
		self.dropout1 = nn.Dropout(0.25)       
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.conv3 = nn.Conv2d(64, 128, 3)
		self.flattern = nn.Flatten()
		self.fc = nn.Linear(832, num_of_labels)
		self.dropout2 = nn.Dropout(0.5)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.pool(x)
		x = self.dropout1(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool(x)
		x = self.dropout1(x)
		x = self.flattern(x)
		x = x.view(-1, 832)
		x = self.fc(x)
		x = self.dropout2(x)
		x = self.softmax(x)
		return x

class Han16(nn.Module):
	def __init__(self):
		super().__init__()
        
		self.conv1_1 = nn.Conv2d(1, 32, 3, padding=2)
		self.conv1_2 = nn.Conv2d(32, 32, 3, padding=2)
		self.conv2_1 = nn.Conv2d(32, 64, 3, padding=2)
		self.conv2_2 = nn.Conv2d(64, 64, 3, padding=2)
		self.conv3_1 = nn.Conv2d(64, 128, 3, padding=2)
		self.conv3_2 = nn.Conv2d(128, 128, 3, padding=2)
		self.conv4_1 = nn.Conv2d(128, 256, 3, padding=2)
		self.conv4_2 = nn.Conv2d(256, 256, 3, padding=2)
		self.pool = nn.MaxPool2d(3, stride=3)
		self.zero_pad = nn.ZeroPad2d(1)
		# self.pool1 = nn.MaxPool2d(x.size(dim=2), x.size(dim=3))
		self.fc1 = nn.Linear(256, 1024)
		self.fc_output = nn.Linear(1024, num_of_labels)
                
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant(m.bias, 0)

	def forward(self, x):
		x = F.leaky_relu(self.conv1_1(self.zero_pad(x)))
		x = self.pool(F.leaky_relu(self.conv1_2(self.zero_pad(x)), negative_slope=0.33))
		x = F.dropout(x, p=0.25)
		x = F.leaky_relu(self.conv2_1(self.zero_pad(x)))
		x = self.pool(F.leaky_relu(self.conv2_2(self.zero_pad(x)), negative_slope=0.33))
		x = F.dropout(x, p=0.25)
		x = F.leaky_relu(self.conv3_1(self.zero_pad(x)))
		x = self.pool(F.leaky_relu(self.conv3_2(self.zero_pad(x)), negative_slope=0.33))
		x = F.dropout(x, p=0.25)
		x = F.leaky_relu(self.conv4_1(self.zero_pad(x)))
		x = F.leaky_relu(self.conv4_2(self.zero_pad(x)), negative_slope=0.33)
		x = F.max_pool2d(x, kernel_size=x.size()[2:]) # global max pooling
		x = x.view(-1, 256)
		x = F.leaky_relu(self.fc1(x), negative_slope=0.33)
		x = F.dropout(x, p=0.5)
		x = self.fc_output(F.sigmoid(x))
		return x

# from resnet_pytorch import ResNet
# model = ResNet.from_pretrained("resnet18")

if len(sys.argv) < 2:
    print("Lack argument")
    exit(1)
if sys.argv[1]=="Net":
    model = Net().to(device)
elif sys.argv[1]=="Han":
    model = Han16().to(device)
elif sys.argv[1]=="Net2":
    model = Net2().to(device)
else:
    print("invalid argument")

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.001)
# optimiser = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
bestloss = 10000
mini_batch = len(train_dataloader)
for epoch in range(30):  # loop over the dataset multiple times
    # for simplicity we only train the model once
	running_loss = 0.0
	for i, (inputs, labels) in enumerate(train_dataloader, start=0):
		# get the inputs; data is a list of [inputs, labels]
		inputs = inputs.unsqueeze(1) # add one dimension
		# since torch.nn.Conv2d() requires 4-dimension input (Batch_size, Channel, x, y)
		# the torch size should thus be [1, 1, 128, 130]
		# for simplicity, batch size is 1

		# zero the parameter gradients
		optimiser.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		_, labels = torch.max(labels, 1)
		loss = criterion(outputs, labels)
		loss.backward()
		optimiser.step()

		# print statistics
		running_loss += loss.item()

		if i % mini_batch == mini_batch-1:  # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
			      (epoch + 1, i + 1, running_loss / mini_batch))
			print(running_loss/mini_batch, bestloss)
			if running_loss/mini_batch < bestloss:
				bestepoch = epoch + 1
				bestloss = running_loss / mini_batch
				bestmodel = model
			running_loss = 0.0
print('Finished Training')
PATH = './wave_model_han16net.pth'
# PATH = './wave_model_net.pth'
print("best loss: ", bestloss, "at epoch", bestepoch)
torch.save(bestmodel.state_dict(), PATH)

bestmodel.load_state_dict(torch.load(PATH))  # 将预训练的参数权重加载到新的模型之中

correct = 0
total = 0
with torch.no_grad():
	for i, (inputs, labels) in enumerate(eval_dataloader, start=0):
		inputs = inputs.unsqueeze(1)  # add one dimension
		outputs = model(inputs)
		newlabels = labels > 0
		indices = newlabels.nonzero()
		_, predicted = torch.max(outputs.data, 1)
		print(predicted)
		labels_ = labels.argmax(dim=1)
		print(labels_)
		total += labels.size(0)
		correct += (predicted == labels_).sum()
		print('label:')
		print(labels)
		print('newlabel:')
		print(indices[0][1])
		print('predicted:')
		print(predicted)
		print('Total: %d' % total)
		print('Correct: %d' % correct)

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))