import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from mpl_toolkits.mplot3d import Axes3D

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader

NITORCH_DIR = os.getcwd()
sys.path.insert(0, NITORCH_DIR)
#nitorch
from nitorch.trainer import Trainer
from nitorch.initialization import weights_init



class syntheticDataset(Dataset):
	'''A dataset of 3D synthetic data '''
	def __init__(self):

		np.random.seed(42)

		n_samples1 = 1000
		mean1 = np.array([0, 0, 0])
		covariance1 = np.array([[10, 0, 0],[0, 10, 0],[0, 0, 5]])
		data_class1 = np.random.multivariate_normal(mean1, covariance1, n_samples1)

		n_samples2 = 750
		mean2 = np.array([-5, 5, -5])
		covariance2 = np.array([[10, 0, 0],[0, 10, 0],[0, 0, 10]])
		data_class2 = np.random.multivariate_normal(mean2, covariance2, n_samples2)

		# Visualize synthetic dataset
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# ax.scatter(data_class1[:,0],data_class1[:,1],data_class1[:,2], c='b', label="Class 0")
		# ax.scatter(data_class2[:,0],data_class2[:,1],data_class2[:,2], c='y', label="Class 1")
		# plt.legend()
		# plt.show()

		self.X = np.vstack([data_class1, data_class2])
		self.y = np.hstack([np.zeros(n_samples1), np.ones(n_samples2)])
	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx].reshape(-1)



# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size).float() 
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, 1).float()   
    
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out


if __name__ == "__main__":

	print("Starting tests ...")
	if(torch.cuda.is_available()):
		device = torch.device('cuda')
		print("Running on GPU ...")
	else:
		device = torch.device('cpu')
		print("Running on CPU ...")
	
	#############################################
	# Test 1 : nitorch.trainer.Trainer()
	#############################################
	print("Testing nitorch.trainer.Trainer() : ")
	
	# DATASET
	BATCH_SIZE = 64
	EPOCHS = 100

	data = syntheticDataset()
	# shuffle and split into test and train
	train_size = int(0.75*len(data))
	train_data, val_data = random_split(
		data, (train_size, len(data) - train_size))

	train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

	# NETWORK
	net = NeuralNet(3, 10).to(device).double()
	net.apply(weights_init)

	criterion = nn.BCEWithLogitsLoss().to(device)
	optimizer = optim.Adam(net.parameters(),
	                       lr=1e-3,
	                       weight_decay=1e-5)
	#TEST
	trainer = Trainer(
	net,
	criterion,
	optimizer,
	# scheduler=None,
	# metrics=metrics,
	# callbacks=callbacks,
	prediction_type="binary")

	# train model and store results
	net, report = trainer.train_model(
	    train_loader,
	    val_loader,
	    device=device,
	    num_epochs=EPOCHS,
	    show_train_steps=50,
	    show_validation_epochs=5
	    )

	#############################################
	# Test 1-a: Check if the loss reduces with   
	# a negative slope on the valdation dataset
	y = (report["val_metrics"]["loss"])
	x = np.arange(len(y))
	slope, intercept, _, _, _ = stats.linregress(x,y)
	# plt.scatter(x, y)
	# plt.plot((slope*x + intercept))
	# plt.show()
	assert slope<0, "Test 1-a Failed - The loss is not reducing with training."