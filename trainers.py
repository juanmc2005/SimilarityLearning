#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from  torch.utils.data import DataLoader
from CenterLoss import CenterLoss
from ContrastiveLoss import ContrastiveLoss
from arcface import ArcLinear
from models import ArcNet
import visual


class BaseTrainer:
    
    def __init__(self, model, device, loss_fn, train_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_batch = len(self.train_loader)
        self.n_train = len(train_loader.dataset)
        self.n_test = len(test_loader.dataset)
        self.losses, self.accuracies = [], []
        self.best_acc = 0
        
    def on_before_epoch(self, epoch):
        """
        Implement custom behavior before the epoch begins
        """
        pass
        
    def feed_forward(self, x, y):
        """
        Compute the output of the model for a mini-batch and return the logits
        """
        raise NotImplementedError("The trainer must implement the method 'feed_forward'")
        
    def embed(self, x):
        """
        Compute the output of the model for a mini-batch and return the embeddings
        """
        raise NotImplementedError("The trainer must implement the method 'embed'")
        
    def get_schedulers(self):
        """
        Return the list of schedulers to use
        """
        raise NotImplementedError("The trainer must implement the method 'get_schedulers'")
        
    def get_optimizers(self):
        """
        Return the list of optimizers to use
        """
        raise NotImplementedError("The trainer must implement the method 'get_optimizers'")
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        """
        Return the desired title for the plot of test embeddings with the best accuracy
        """
        raise NotImplementedError("The trainer must implement the method 'get_best_acc_plot_title'")
        
    def train(self, epochs=10, log_interval=20):
        for sch in self.get_schedulers():
            sch.step()
        for i in range(1, epochs+1):
            self.on_before_epoch(i)
            self.train_epoch(i, log_interval)
        
    def train_epoch(self, epoch, log_interval):
        total_loss, correct = 0, 0
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Feed Forward
            logits = self.feed_forward(x, y)
            loss = self.loss_fn(logits, y)
            
            # Backprop
            optims = self.get_optimizers()
            for op in optims:
                op.zero_grad()
            loss.backward()
            for op in optims:
                op.step()
            
            # Loss and Accuracy tracking
            total_loss += loss
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == y.data).sum()
            
            # Logging
            if i % log_interval == 0:
                print(f"Train Epoch: {epoch} [{100. * i / self.n_batch:.0f}%]\tLoss: {loss.item():.6f}")
        
        test_correct = self.test()
        loss = total_loss / self.n_train
        acc = 100 * test_correct / self.n_test
        self.losses.append(loss)
        self.accuracies.append(acc)
        print(f"--------------- Epoch {epoch} Results ---------------")
        print(f"Training Accuracy = {100 * correct / self.n_train:.0f}%")
        print(f"Mean Training Loss: {loss:.6f}")
        print(f"Test Accuracy: {test_correct} / {self.n_test} ({acc:.0f}%)")
        print("-----------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            self.visualize(self.test_loader, self.get_best_acc_plot_title(epoch, acc), plot_name)
    
    def test(self):
        correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.feed_forward(x, y)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y.data).sum()
        return correct
    
    def visualize(self, loader, title, filename):
        embs, targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                embs.append(self.embed(x))
                targets.append(y)
        embs = torch.cat(embs, 0).float().data.cpu().numpy()
        targets = torch.cat(targets, 0).float().cpu().numpy()
        visual.visualize(embs, targets, title, filename)


class CenterTrainer:
    
    def __init__(self, model, device, loss_weight=1):
        self.loss_weight = loss_weight
        self.device = device
        self.model = model.to(device)
        self.nllloss = nn.NLLLoss().to(device)
        self.centerloss = CenterLoss(10, 2).to(device)
        self.optimizer4nn = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        self.optimzer4center = optim.SGD(self.centerloss.parameters(), lr=0.5)
    
    def train(self, epoch, loader):
        print("Training... Epoch = %d" % epoch)
        ip1_loader = []
        idx_loader = []
        for i, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
    
            ip1, pred = self.model(data)
            loss = self.nllloss(pred, target) + self.loss_weight * self.centerloss(target, ip1)
    
            self.optimizer4nn.zero_grad()
            self.optimzer4center.zero_grad()
    
            loss.backward()
    
            self.optimizer4nn.step()
            self.optimzer4center.step()
    
            ip1_loader.append(ip1)
            idx_loader.append((target))
    
        feat = torch.cat(ip1_loader, 0)
        labels = torch.cat(idx_loader, 0)
        visual.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(),
                         "Epoch = {}".format(epoch), "epoch={}".format(epoch))


class ArcTrainerBetter(BaseTrainer):

    def __init__(self, trainset, testset, device, nfeat, nclass, margin=0.2, s=7.0, batch_size=100):
        train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)
        super(ArcTrainerBetter, self).__init__(
                ArcNet(),
                device,
                nn.CrossEntropyLoss().to(device),
                train_loader,
                test_loader)
        self.arc = ArcLinear(nfeat, nclass, margin, s).to(device)
        self.margin = margin
        self.s = s
        self.optimizers = [
                optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005),
                optim.SGD(self.arc.parameters(), lr=0.01)
        ]
        self.schedulers = [
                lr_scheduler.StepLR(self.optimizers[0], 20, gamma=0.5)
        ]
    
    def feed_forward(self, x, y):
        feat = self.model(x)
        logits = self.arc(feat, y)
        return logits
        
    def embed(self, x):
        return self.model(x)
        
    def get_schedulers(self):
        return self.schedulers
        
    def get_optimizers(self):
        return self.optimizers
    
    def get_best_acc_plot_title(self, epoch, accuracy):
        return f"Test Embeddings (Epoch {epoch}) - {accuracy:.0f}% Accuracy - m={self.margin} s={self.s}"


# TODO remove when confirmed that the other version works well
class ArcTrainer:
    
    def __init__(self, model, device, nfeat, nclass, margin=0.2, s=7.0):
        self.margin = margin
        self.s = s
        self.device = device
        self.model = model.to(device)
        self.arc = ArcLinear(nfeat, nclass, margin, s).to(device)
        self.lossfn = nn.CrossEntropyLoss().to(device)
        self.optim_nn = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        self.optim_arc = optim.SGD(self.arc.parameters(), lr=0.01)
        self.sheduler = lr_scheduler.StepLR(self.optim_nn, 20, gamma=0.5)
        self.losses, self.accuracies = [], []
        self.best_acc = 0
    
    def train(self, epoch, loader, test_loader, log_interval=20):
        self.sheduler.step()
        test_total = len(test_loader.dataset)
        total_loss, correct, total = 0, 0, 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # Feed Forward
            feat = self.model(x)
            logits = self.arc(feat, y)
            loss = self.lossfn(logits, y)
            
            total_loss += loss
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y.data).sum()
            
            # Backprop
            self.optim_nn.zero_grad()
            self.optim_arc.zero_grad()
            
            loss.backward()
            
            self.optim_nn.step()
            self.optim_arc.step()
            
            # Logging
            if i % log_interval == 0:
                print(f"Train Epoch: {epoch} [{100. * i / len(loader):.0f}%]\tLoss: {loss.item():.6f}")
        
        test_correct = self.eval(test_loader)
        loss = total_loss / len(loader)
        acc = 100 * test_correct / test_total
        self.losses.append(loss)
        self.accuracies.append(acc)
        print(f"--------------- Epoch {epoch} Results ---------------")
        print(f"Training Accuracy = {100 * correct / total:.0f}%")
        print(f"Mean Training Loss: {loss:.6f}")
        print(f"Test Accuracy: {test_correct} / {test_total} ({acc:.0f}%)")
        print("-----------------------------------------------")
        if test_correct > self.best_acc:
            plot_name = f"test-feat-epoch-{epoch}"
            print(f"New Best Test Accuracy! Saving plot as {plot_name}")
            self.best_acc = test_correct
            self.visualize(test_loader, f"Test Embeddings (Epoch {epoch}) - {acc:.0f}% Accuracy - m={self.margin} s={self.s}", plot_name)
        
    def eval(self, loader):
        correct = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                feat = self.model(x)
                logits = self.arc(feat, y)
                _, predicted = torch.max(logits.data, 1)
                correct += (predicted == y.data).sum()
        return correct
    
    def visualize(self, loader, title, filename):
        embs, targets = [], []
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device), target.to(self.device)
                embs.append(self.model(x))
                targets.append(target)
        embs = torch.cat(embs, 0).type(torch.FloatTensor).data.cpu().numpy()
        targets = torch.cat(targets, 0).type(torch.FloatTensor).cpu().numpy()
        visual.visualize(embs, targets, title, filename)


class ContrastiveTrainer:
    
    def __init__(self, model, device, margin=1.0, distance='euclidean'):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = ContrastiveLoss(margin, distance).to(device)
        self.optimizer = optim.Adam(model.parameters())
        self.sheduler = lr_scheduler.StepLR(self.optimizer,20,gamma=0.8)
        self.losses, self.accuracies = [], []
    
    def train(self, epoch, loader, test_loader, visu_loader, report_interval=1000):
        print("[Epoch %d]" % epoch)
        self.sheduler.step()
        running_loss = 0.0
        total_loss = 0.0
        best_acc = 0.0
        for i, (data1, data2, y, _, _) in enumerate(loader):
            data1, data2, y = data1.to(self.device), data2.to(self.device), y.to(self.device)
            
            emb1 = self.model(data1)
            emb2 = self.model(data2)
            loss = self.loss_fn(y, emb1, emb2)
            running_loss += loss
            total_loss += loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % report_interval == report_interval-1:
                print("[%d batches] Loss = %.4f" % (i+1, running_loss / report_interval))
                running_loss = 0.0
        
        acc = self.eval(test_loader)
        loss = total_loss / len(loader)
        self.losses.append(loss)
        self.accuracies.append(acc)
        print("Training Loss: %.4f" % loss)
        print("Test Accuracy: {}%".format(acc))
        if acc > best_acc:
            best_acc = acc
            self.visualize(visu_loader, "epoch={}-acc={}".format(epoch, acc))
    
    def eval(self, loader):
        correct, total = 0.0, 0.0
        with torch.no_grad():
            for x1, x2, y, _, _ in loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                emb1 = self.model(x1)
                emb2 = self.model(x2)
                dist = self.loss_fn.distance(emb1, emb2)
                preds = torch.where(dist < self.loss_fn.margin, torch.zeros_like(dist), torch.ones_like(dist))
                correct += (preds == y).sum()
                total += y.size(0)
        return 100 * correct / total
    
    def visualize(self, loader, filename):
        embs, targets = [], []
        with torch.no_grad():
            for x, target in loader:
                x, target = x.to(self.device), target.to(self.device)
                embs.append(self.model(x))
                targets.append(target)
        embs = torch.cat(embs, 0).type(torch.FloatTensor).data.cpu().numpy()
        targets = torch.cat(targets, 0).type(torch.FloatTensor).cpu().numpy()
        visual.visualize(embs, targets, "Test Embeddings", filename)
                
    
    def train_online_recomb(self, epoch, loader, xtest, ytest):
        print("Training... Epoch = %d" % epoch)
        for i, (x, y) in enumerate(loader):
            x, y = x.to(self.device).unsqueeze(1), y.to(self.device)
            x1, x2, simil = self.build_pairs(x, y)
            x1, x2, simil = x1.to(self.device), x2.to(self.device), simil.to(self.device)
            emb1 = self.model(x1)
            emb2 = self.model(x2)
            loss = self.loss_fn(simil, emb1, emb2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        embs = self.model(xtest).data.cpu().numpy()
        visual.visualize(embs, ytest.cpu().numpy(), "Epoch = {}".format(epoch), "epoch={}".format(epoch))
    
    def build_pairs(self, X, Y):
        n = Y.size(0)
        fst, snd = [], []
        Ypairs = []
        for i in range(n-1):
            added = []
            for j in range(i+1, n):
                label = Y[j]
                if i != j and label not in added:
                    fst.append(X[i])
                    snd.append(X[j])
                    Ypairs.append(0 if Y[i] == Y[j] else 1)
                    added.append(label)
                if len(added) == 10:
                    break
        pairs1 = torch.cat(fst, 0).type(torch.FloatTensor)
        pairs2 = torch.cat(snd, 0).type(torch.FloatTensor)
        return pairs1, pairs2, torch.FloatTensor(Ypairs)
