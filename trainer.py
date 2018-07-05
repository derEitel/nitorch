import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from nitorch.callbacks import ModelCheckpoint

def train_model(net, criterion, optimizer, scheduler,
                train_loader, val_loader, 
                train_sampler, val_sampler,
                gpu=0, num_epochs=25,
                show_train_steps=25,
                show_validation_epochs=1,
                metrics=[], callbacks=[],
                class_threshold=0.5):
    """ Main function to train a network."""

    total_loss = []
    val_total_loss = []

    val_metrics = dict()
    train_metrics = dict()
    
    start_time = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        scheduler.step(epoch)
        # variables to compute metrics
        all_preds = []
        all_labels = []
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
            loss = criterion(outputs, labels) # BCEWithlogits computs sigmoid itself
            loss.backward()
            optimizer.step()
            #print_gradients(net)
            
            # store results
            sigmoid = F.sigmoid(outputs)
            predicted = sigmoid.data >= class_threshold
            for j in range(len(predicted)):
                all_preds.append(predicted[j].cpu().numpy()[0])
                all_labels.append(labels[j].cpu().numpy()[0])

                
            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % show_train_steps == 0:    # print every X mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / show_train_steps))
                running_loss = 0.0                
        
        # report training metrics after each epoch
        train_metrics = report_metrics(train_metrics, metrics, all_labels, all_preds, phase="Train")

        epoch_loss /= len(train_loader)
        total_loss.append(epoch_loss)
        
        # validate every x iterations
        if epoch % show_validation_epochs == 0:
            net.eval()
            validation_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data['image'], data['label']
                    # wrap data in Variable
                    inputs, labels = Variable(inputs.cuda(gpu)), Variable(labels.cuda(gpu))
                    # forward pass only
                    outputs = net(inputs)            
                    # compute validation accuracy
                    sigmoid = F.sigmoid(outputs)
                    predicted = sigmoid.data >= class_threshold
                    loss = criterion(outputs, labels)
                    for j in range(len(predicted)):
                        all_preds.append(predicted[j].cpu().numpy()[0])
                        all_labels.append(labels[j].cpu().numpy()[0])

                    
                    validation_loss += loss.item()

                val_metrics = report_metrics(val_metrics, metrics, all_labels, all_preds, phase="Val")
                validation_loss /= len(val_loader)
                print("Val loss: {0:.4f}".format(validation_loss))
                val_total_loss.append(validation_loss)
        if callbacks is not None:
            for callback in callbacks:
                callback(epoch, net, val_metrics)

    total_time = ((time.time() - start_time) / 60)
    print("Time trained: {:.2f} minutes".format(total_time))
    
    # execute final methods of callbacks
    if callbacks is not None:
        for callback in callbacks:
            # find all methods of the callback
            method_list = [func for func in dir(callback) if callable(getattr(callback, func)) and not func.startswith("__")]
            if "final" in method_list:
                # in case of model checkpoint load best model
                if isinstance(callback, ModelCheckpoint):
                    best_acc, best_model = callback.final()
                    net.load_state_dict(best_model)
                else:
                    callback.final()
    return net, best_acc, [total_loss, val_total_loss, train_metrics, val_metrics]

def visualize_training(report, metrics=[]):
    plt.figure()
    plt.plot(report[0])
    plt.plot(report[1])
    plt.title("Loss during training")
    plt.legend(["Train", "Val"])
    plt.show()
    
    for metric in metrics:
        plt.figure()
        plt.plot(report[2][metric.__name__])
        plt.plot(report[3][metric.__name__])
        plt.legend(["Train", "Val"])
        plt.title(metric.__name__)
        plt.show()

def evaluate_model(net, val_loader, gpu=0):
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

def report_metrics(metrics_dict, metrics, all_labels, all_preds, phase):
    """ Function to compute a list of metric functions. """
    for metric in metrics:
        result = metric(all_labels, all_preds)
        if metric.__name__ in metrics_dict:
            metrics_dict[metric.__name__].append(result)
        else:
            metrics_dict[metric.__name__] = [result]
        # print result
        if isinstance(result, float):
            print("{} {}: {:.2f} %".format(phase, metric.__name__, result * 100))
        else:
            print("{} {}: {} ".format(phase, metric.__name__, str(result)))        
    return metrics_dict
