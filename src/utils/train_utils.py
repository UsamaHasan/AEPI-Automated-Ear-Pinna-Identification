import torch
from tqdm.notebook import tqdm
def train_model(model ,optimizer, schedular ,criterion,binary_criterion, NUM_EPOCHS):
    since = time.time()
    best_acc = 0.0
    for epoch in tqdm(range(NUM_EPOCHS)):
        print(f'EPOCH{epoch}/{NUM_EPOCHS-1}')
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
                running_loss = 0.0
                running_correct = 0
                class_correct = 0.0
                binary_correct = 0.0
                total = 0
                for images , labels in tqdm((iter(trainLoader))):
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    #Forward Pass
                    class_logits , binary_logits = model(images,labels[:,:1])
                    #Calculate loss   
                    class_labels = torch.squeeze(labels[:,:1])
                    binary_labels = torch.squeeze(labels[:,1:]).float()
          
                    class_loss = criterion(class_logits,class_labels)
                    
                    binary_loss = binary_criterion(torch.squeeze(binary_logits),binary_labels)                     
                    _ , class_preds = torch.max(class_logits ,1)
                    binary_preds = torch.round(binary_logits)
                    
                    # Add loss of both softmax loss and binary loss
                    loss = class_loss + binary_loss
                    #Calculate grads
                    loss.backward()
                    optimizer.step()
                    if epoch > 30:
                        schedular.step()
                    running_loss += loss.item() 
                    total += labels.size(0)
                    class_correct += (class_preds == class_labels).sum().item()
                    binary_correct += (torch.squeeze(binary_preds) == binary_labels).sum().item()
                    
                schedular.step()
                print(f'Train Epoch:{epoch} Loss:{running_loss / len(trainLoader)}  Accuracy:{100 * class_correct / total} Gender Accuracy:{100 * binary_correct / total}')

            else:
                top3_acc.reset()            
                model.eval()
                running_loss = 0.0
                running_correct = 0
                class_correct = 0.0
                binary_correct = 0.0
                total = 0
                for images , labels in tqdm(iter(validationLoader)):
                    images = images.to(device)
                    labels = labels.to(device)
                    class_logits , binary_logits= model(images,labels[:,:1])
                    class_labels = torch.squeeze(labels[:,:1])
                    binary_labels = torch.squeeze(labels[:,1:]).float()

                    class_loss = criterion(class_logits,class_labels)
                    binary_loss = binary_criterion(torch.squeeze(binary_logits),binary_labels)
                    loss = class_loss + binary_loss
                    _ , class_preds = torch.max(class_logits ,1)
                    binary_preds = torch.round(binary_logits)
                    running_loss += loss.item()
                    total += labels.size(0)
                    top3_acc.update(class_logits, class_labels)
                    
                    class_correct += (class_preds == class_labels).sum().item()
                    binary_correct += (torch.squeeze(binary_preds) == binary_labels).sum().item()
                print(f'Eval Epoch:{epoch} Loss:{running_loss / len(validationLoader)} Accuracy:{100 * class_correct / total } Gender Accuracy:{100 * binary_correct / total}')
                print(f'{top3_acc.name}: {top3_acc.get()*100} ')
             
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model

def init_weights(m):
    """"""
    if isinstance(m,(nn.Conv2d ,nn.Linear )) :
        torch.nn.init.kaiming_normal_(m.weight)
