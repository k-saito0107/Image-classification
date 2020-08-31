import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


def train(model, num_epochs,train_loader, test_loader):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logs = []

    for epoch in range(1, num_epochs+1):
        for img, label in train_loader:
            model.train()
            img , label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            t_loss = criterion(outputs, label)
            t_loss.backward()
            optimizer.step()
        
        if epoch%10 == 0:
            if epoch == 0:
                continue
            print('Epoch {}/{}'.format(epoch,num_epochs))
            correct = 0 #正解したデータの総数
            total = 0 #予測したデータの総数
            running_loss = 0.0
            for v_img, v_label in test_loader:
                model.val()
                v_img , v_label = v_img.to(device), v_label.to(device)
                v_outputs = model(v_img)
                v_loss=criterion(v_outputs,v_label)
                running_loss += v_loss.item()
                _, predicted = torch.max(v_outputs.data, 1)
                total += v_label.size(0)
                # 予測したデータ数を加算
                correct += (predicted == v_label).sum().item()
            val_acc=correct/total
            val_loss = running_loss/len(test_loader)
            print('train_loss : {},  val_loss : {},  val_acc : {}'.format(t_loss, val_loss, val_acc))

            #ログを保存
            log_epoch = {'epoch' : epoch, 'train_loss' : t_loss, 'val_loss' : val_loss,'val_acc' : val_acc}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv('kw_resources/画像分類/log_out.csv')
        
        if epoch % 100 == 0 and epoch != 10:
            print('---------------------------------------------------------------')
        
    
    return model





