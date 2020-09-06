import torch
from tqdm import tqdm

def train_model(model , optimizer ,criterion, NUM_EPOCHS):
    """
    """
    since = time.time()
    best_acc = 0.0
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f'EPOCH{epoch}/{NUM_EPOCHS-1}')
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
                running_loss = 0.0
                running_correct = 0
                for images , labels in (iter(trainLoader)):
                    labels = torch.LongTensor(labels)
                    outputs = torch.zeros(size=(len(images),164))
                    for i ,image in enumerate(images):
                        # Inserting Batch size 1 in image
                        image = image.to(device)
                        image = image.view(1,image.shape[0],image.shape[1],image.shape[2])   
                        output = model(image)
                        outputs[i] = output   
                    loss = criterion(outputs,labels)    
                    _ , preds = torch.max(outputs ,1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * len(images)
                    running_correct += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(x_train)
                epoch_acc = running_correct.double() / len(x_train)
                print(f'Train: Epoch{epoch}  Loss:{epoch_loss} ,  Accuracy{epoch_acc}')  
            else:
                model.eval()
                running_loss = 0.0
                running_correct = 0
                for images , labels in iter(testLoader):
                    labels = torch.LongTensor(labels)
                    outputs = torch.zeros(size=(len(images),164))
                    for i ,image in enumerate(images):
                        # Inserting Batch size 1 in image
                        image = image.to(device)
                        image = image.view(1,image.shape[0],image.shape[1],image.shape[2])   
                        output = model(image)
                        print(output)
                        outputs[i] = output

                    _ , preds = torch.max(outputs ,1)
                    loss = criterion(outputs , labels)
                    running_loss += loss.item() * len(images)
                    running_correct += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(x_test)
                epoch_acc = running_correct.double() / len(x_test)
                print(f'Test: Epoch:{epoch}, Loss:{epoch_loss}, Accuracy:{epoch_acc}')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_weights = copy.deepcopy(model.state_dict())  
            print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_weights)
    return model

def init_weights(m):
    """"""
    if isinstance(m,(nn.Conv2d ,nn.Linear )) :
        torch.nn.init.kaiming_normal_(m.weight)