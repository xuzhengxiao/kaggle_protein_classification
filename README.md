This project is code of Human Protein Atlas Image Classification competition<br>
## Getting Started
* [dataset.py](https://github.com/xuzhengxiao/sgdr_rgb/blob/master/dataset.py) is self customed data loader solution independent of Deep learning frameworks.When applied in ensemble model,the last model's memory can't be released completely,thus causing increasing memeory usage.So I finally replace it with pytorch's data loader in [pytorch_dataset.py](https://github.com/xuzhengxiao/sgdr_rgb/blob/master/pytorch_dataset.py).<br>
* [resnet.py](https://github.com/xuzhengxiao/sgdr_rgb/blob/master/resnet.py) trains model.<br>
* [transforms.py](https://github.com/xuzhengxiao/sgdr_rgb/blob/master/transforms.py). image augmentation.<br>
* [utils.py](https://github.com/xuzhengxiao/sgdr_rgb/blob/master/utils.py) handles csv file,calculates class weight and splits train and validation data.
* [predict.py](https://github.com/xuzhengxiao/sgdr_rgb/blob/master/predict.py) predicts test data

## Some appcoaches taken 
*learning rate schedule* is warm restarts,which is described in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983).<br>
*bce loss* combined with *log-dampned weight* works better.<br>

 ## Attention
 * If your hardware can't deal with large batch,you can try accumulating gradient tech.<br>
 ```
 for iteration,(images,labels) in enumerate(train_loader):
    images = images.cuda()
    labels = labels.cuda()
    inputs = Variable(images)
    targets = Variable(labels)
    outputs=model(inputs)
    loss=criterion(outputs,targets)
    # loss regularization
    loss = loss/accumulation_steps   
    loss.backward()
    if(iteration%accumulation_steps)==0:
        optimizer.step()        # update parameters of net
        optimizer.zero_grad()   # reset gradient
 ```
 * **we trained our model only in train data,no external data nor leaked data.** External data can boost about 0.1 in lb.However,someone in competition's discussion claimed that he could achieve 0.6 without external data,so we didn't take external data into consideration.<br>
 * our final result ends with public lb 0.474,private lb 0.443,wihch is obvious not good.Because this is our first competition,we do not konw lots of techniques useful in such competition.We still have a long way to go. 
