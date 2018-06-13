import time
from copy import deepcopy
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def train_model(net, criterion, optimizer, scheduler,
                train_loader, val_loader, 
                train_sampler, val_sampler,
                gpu=0, num_epochs=25, ignore_epochs=-1,
                show_train_steps=25):
    """ Main function to train a network."""

    total_loss = []
    val_total_loss = []
    val_acc = []
    train_acc = []
    
    start_time = time.time()
    best_acc = 0.0
    best_model_wts = deepcopy(net.state_dict())
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        train_correct = 0
        scheduler.step(epoch)
        # train
        net.train()        
        for i, data in enumerate(train_loader):
            inputs, labels = data['image'], data['label']
            # wrap data in Variable
            inputs, labels = Variable(inputs.cuda(gpu)), Variable(labels.cuda(gpu))
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print_gradients(net)
            
            # compute training accuracy
            sigmoid = F.sigmoid(outputs)
            for j, out in enumerate(sigmoid):
                cls = 0 if out.data[0] < 0.5 else 1
                if cls  == data['label'][j][0]:
                    train_correct += 1
                
            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % show_train_steps == 0:    # print every X mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / show_train_steps))
                running_loss = 0.0                
        
        # report training accuracy after each epoch
        acc = (train_correct / len(train_sampler)) * 100
        train_acc.append(acc)
        print("Train acc: {0:.2f} %".format(acc))

        # validate every x iterations
        if epoch % 1 == 0:
            net.eval()
            validation_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data['image'], data['label']
                    # wrap data in Variable
                    inputs, labels = Variable(inputs.cuda(gpu)), Variable(labels.cuda(gpu))
                    # forward pass only
                    outputs = net(inputs)                
                    # compute validation accuracy
                    sigmoid = F.sigmoid(outputs)
                    for j, out in enumerate(sigmoid):
                        cls = 0 if out.data[0] < 0.5 else 1
                        if cls  == data['label'][j][0]:
                            val_correct += 1
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()
                acc = (val_correct / len(val_sampler)) * 100
                # store accuracy and save model weights if best
                val_acc.append(acc)
                if acc > best_acc and epoch > ignore_epochs:
                    best_acc = acc
                    best_model_wts = deepcopy(net.state_dict())

                print("Val acc: {0:.2f} %".format(acc))
                validation_loss /= len(val_loader)
                print("Val loss: {0:.4f}".format(validation_loss))
                val_total_loss.append(validation_loss)

        epoch_loss /= len(train_loader)
        total_loss.append(epoch_loss)

    total_time = ((time.time() - start_time) / 60)
    print("Time trained: {:.2f} minutes".format(total_time))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net, best_acc, [total_loss, val_total_loss, train_acc, val_acc]

def visualize_training(report):
    plt.figure()
    plt.plot(report[0])
    plt.plot(report[1])
    plt.title("Loss during training")
    plt.legend(["Train", "Val"])
    plt.show()
    
    plt.figure()
    plt.plot(report[2])
    plt.plot(report[3])
    plt.legend(["Train", "Val"])
    plt.title('Accuracy')
    plt.show()

def evaluate_model(net, val_loader):
    # predict on the validation set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data['image'], data['label']
            # wrap data in Variable
            inputs, labels = Variable(inputs.cuda(gpu)), Variable(labels.cuda(gpu))
            # forward + backward + optimize
            outputs = net(inputs)
            sigmoid = F.sigmoid(outputs)
            for j, out in enumerate(sigmoid):
                cls = 0 if out.data[0] < 0.5 else 1
                all_preds.append(cls)
                all_labels.append(data['label'][j][0])
            
    # compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Visualize the confusion matrix
    classes = ["control", "patient"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()