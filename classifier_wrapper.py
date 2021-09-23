from bbox import AbstractBBox
import torch
import numpy as np

# wrapper per random forest etc
class sklearn_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier):
        super().__init__()
        self.bbox = classifier

    def model(self):
        return self.bbox

    def predict(self, X):
        return self.bbox.predict(X)

    def predict_proba(self, X):
        return self.bbox.predict_proba(X)


# da aggiungere wrapper per altri metodi in pytorch
class pytorch_classifier_wrapper(AbstractBBox):
    def __init__(self, classifier, optimizer, loss_function, classes):
        super().__init__()
        self.bbox = classifier
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.classes_ = classes

    def model(self):
        return self.bbox  # ritorna il modello in pytorch

    # def state_dict(self):
    #   FILE = "ann.pth"
    #  return torch.save(self.bbox.state_dict(), FILE)
    def evaluate(self, X):
        y_pred_list = []
        vett = []
        with torch.no_grad():
            self.bbox.eval()
            for x, y in X:
                # X_batch = X_batch.to(device)
                y_test_pred = self.bbox(x.float())
                y, y_pred_tags = torch.max(y_test_pred, dim=1)
                y_pred_list.append(y_pred_tags.numpy())
        print(len(y_pred_list))
        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        print("LEN X ", x.shape)
        for el in y_pred_list:
            for elemento in el:
                vett.append(elemento)
        print("LEN vett ", len(vett))
        return vett

    def fit(self, X_train, y_train):
        train_target = torch.tensor(list(X_train), dtype=torch.float64)
        train = torch.tensor(list(y_train), dtype=torch.float64)
        train_tensor = torch.utils.data.TensorDataset(train, train_target)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_tensor, batch_size=64, shuffle=True
        )
        for _ in range(50):  # 200 full passes over the data
            for data in train_loader:  # is a batch of data
                X, y = data  # X is the batch of features, y is the batch of targets.
                X = X.float().view(-1, 236)
                y = y.float()
                self.bbox.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                output = self.bbox(
                    X
                )  # pass in the reshaped batch (recall they are 1*237) view=shape
                loss = self.loss_function(
                    output, y.long()
                )  # calc and grab the loss value
                loss.backward()  # apply this loss backwards thru the network's parameters
                self.optimizer.step()  # attempt to optimize weights to account for loss/gradients

    def predict(self, X):
        self.bbox.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            output = self.bbox(X_tensor)
            y_pred = torch.argmax(output, dim=1)
        return np.array(y_pred)  # ritorna la classe

    def predict_proba(self, X):
        """self.bbox.eval() 
        #y_pred_list=[]
        with torch.no_grad():
            X=X.view(-1,600)
            y_pred=self.bbox(X)
        #y_pred_list = [a.squeeze().tolist() for a in y_pred]
        return y_pred #ritorna il vetore delle probabilità, thanks to Soft max layer"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            y_prob = self.bbox(X_tensor)
        return np.array(y_prob)

    def predict_proba1(self, X):
        self.bbox.eval()
        # y_pred_list=[]
        with torch.no_grad():
            X = X.view(-1, 600)
            y_pred = self.bbox(X)
        # y_pred_list = [a.squeeze().tolist() for a in y_pred]
        return y_pred  # ritorna il vetore delle probabilità, thanks to Soft max layer


""" Predict_proba with PyTorch 
net = torch.load('resnet50_trained3.pth')
print(net)
net.eval()
output = net(image)
print(output) #print output from crossentropy score
sm = torch.nn.Softmax()
probabilities = sm(output) 
print(probabilities) #Converted to probabilities

OPPURE
model.eval()
logits = model(data)
probs = F.softmax(logits, dim=1) # assuming logits has the shape [batch_size, nb_classes]
preds = torch.argmax(logits, dim=1)"""
