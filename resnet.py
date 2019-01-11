import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet34
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_dataset import ProteinDataset
import utils
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import f1_score
from collections import OrderedDict
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# cacluate log-dampned weight
#_,weight=utils.create_class_weight()

weight={0: 1.0, 1: 3.01, 2: 1.95, 3: 2.79, 4: 2.61, 5: 2.31, 6: 3.23, 7: 2.2, 8: 6.17, 9: 6.34, 10: 6.81,
 11: 3.15, 12: 3.61, 13: 3.86, 14: 3.17, 15: 7.1, 16: 3.87, 17: 4.8, 18: 3.34, 19: 2.84, 20: 4.99,
 21: 1.91, 22: 3.46, 23: 2.15, 24: 4.37, 25: 1.13, 26: 4.35, 27: 7.74}
weight=list(weight.values())

def build_parser():
  parser = ArgumentParser()

  parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
  parser.add_argument('--epochs', type=int, default=36, help='number of epochs to train for')
  return parser

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super(FocalLoss,self).__init__()
#         self.gamma = gamma
#
#     def forward(self, input, target):
#         if not (target.size() == input.size()):
#             raise ValueError("Target size ({}) must be the same as input size ({})"
#                              .format(target.size(), input.size()))
#
#         max_val = (-input).clamp(min=0)
#         loss = input - input * target + max_val + \
#                ((-max_val).exp() + (-input - max_val).exp()).log()
#
#         invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
#         loss = (invprobs * self.gamma).exp() * loss
#
#         return loss.sum(dim=1).mean()



class FocalLoss(nn.Module):

  def __init__(self, gamma=2, weight=None):
    super(FocalLoss, self).__init__()
    self.gamma = gamma
    self.weight = weight

  def forward(self, input, target):

    weight = torch.FloatTensor(self.weight).cuda()

    # compute the negative likelyhood
    logpt = - F.binary_cross_entropy_with_logits(input, target, weight=weight,reduce=False)
    pt = torch.exp(logpt)

    # compute the loss
    focal_loss = -((1 - pt) ** self.gamma) * logpt
    #balanced_focal_loss = self.balance_param * focal_loss
    return torch.mean(focal_loss)

def acc(y_true, y_pred, threshold=0.5):
  y_pred=y_pred.sigmoid()
  y_pred=(y_pred>threshold).float()
  return torch.mean((y_true==y_pred).float())

class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class Resnet(nn.Module):
    def __init__(self, pre=True):
        super(Resnet,self).__init__()
        encoder = resnet34(pretrained=pre)

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1, self.bn1,self.relu, self.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        # head
        self.head=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )
        self.fc=nn.Linear(512, 28)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def evaluate(model,criterion,val_loader):

  accs=AverageMeter()
  losses=AverageMeter()
  error_scores=AverageMeter()
  model.eval()
  with torch.no_grad():
    for i,(images,labels) in enumerate(val_loader):
      images = images.cuda()
      labels = labels.cuda()
      inputs = Variable(images)
      targets = Variable(labels)
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      losses.update(loss.item(),inputs.size(0))
      accs.update(acc(targets.data.cpu(),outputs.data.cpu()),inputs.size(0))
      if i==0:
          total_outputs=outputs
          total_targets=targets
      else:
          total_outputs=torch.cat([total_outputs,outputs],0)
          total_targets=torch.cat([total_targets,targets],0)
      error_scores.update(f1_score(targets.data.cpu(),outputs.sigmoid().data.cpu() > 0.5,average='macro'),inputs.size(0))
    f1=f1_score(total_targets.data.cpu(),total_outputs.sigmoid().data.cpu() > 0.5,average='macro')
  return losses.avg,accs.avg,f1,error_scores.avg

def set_optimizer_lr(optimizer, lr):
  # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return optimizer

def train(model,epochs,train_loader,val_loader,model_path='best.pkl'):
    min_lr = 0
    max_lr = 0.1
    steps_per_epoch = len(train_loader)
    lr_decay = 1
    cycle_length = 2
    mult_factor = 2
    batch_since_restart = 0
    next_restart = cycle_length

    optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=1e-4)
    #criterion = FocalLoss().cuda()
    criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(weight)).cuda()
    #criterion=FocalLoss(weight=weight).cuda()
    best_score=-1

    for epoch in range(epochs):
      model.train()
      print ("Epoch {}/{}".format(epoch+1,epochs))
      for iteration,(images,labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        inputs = Variable(images)
        targets = Variable(labels)
        ########################### this bug
        if iteration==steps_per_epoch-1:
          print ('{}/{} - last iter skip to avoid bug'.format(iteration+1,steps_per_epoch))
          batch_since_restart += 1
          continue
        outputs=model(inputs)
        loss=criterion(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("{}/{} - loss: {:.4f} - acc: {:.4f}".format(1+iteration,steps_per_epoch, loss.item(),acc(targets.data.cpu(),outputs.data.cpu())))

        batch_since_restart += 1
        fraction_to_restart = batch_since_restart / (steps_per_epoch * cycle_length)
        clr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        optimizer = set_optimizer_lr(optimizer, clr)


      """ evaluate model """
      val_loss,val_acc,val_score,error_score=evaluate(model,criterion,val_loader)
      print ("val_loss: {:.4f} - val_acc: {:.4f} -  val_score: {:.4f} - error_score: {:.4f}".format(val_loss, val_acc,val_score,error_score))
      if val_score>best_score:
        if epoch==0:
          print ("Epoch {:05d}: val_score improved from {} to {:.4f}, saving model to {}".format(epoch + 1, 'inf', val_score, model_path))
        else:
          print ("Epoch {:05d}: val_score improved from {:.4f} to {:.4f}, saving model to {}".format(epoch+1,best_score,val_score,model_path))
        best_score=val_score
        torch.save(model.state_dict(),model_path)
      else:
        print ("Epoch {:05d}: val_score did not improve from {:.4f}".format(epoch+1,best_score))

      ########################### warm restart
      if epoch + 1 == next_restart:
        batch_since_restart = 0
        cycle_length = np.ceil(cycle_length * mult_factor)
        next_restart += cycle_length
        max_lr *= lr_decay

    torch.save(model.state_dict(),'final_epoch.pkl')



def main():
    parser = build_parser()
    options = parser.parse_args()
    batch_size=options.batch_size
    #train_names=utils.train_names

    # train_set = utils.get_train()
    # val_set = utils.val_n

    model = Resnet()
    model.cuda()
    model = torch.nn.DataParallel(model)

    ###########################
    #train_set,val_set = train_test_split(train_names, test_size=0.2, random_state=2050)
    train_set,val_set=utils.get_split()
    train_set = utils.get_train(train_set)

    train_datasest=ProteinDataset(dirpath=utils.TRAIN,fnames=train_set)
    train_loader = DataLoader(train_datasest, batch_size=batch_size, shuffle=True,num_workers=4)

    val_dataset = ProteinDataset(dirpath=utils.TRAIN,fnames=val_set)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train(model, options.epochs, train_loader,val_loader,'sgdr_rgb.pkl')

    # splitter = RepeatedKFold(n_splits=5, n_repeats=1, random_state=2050)
    # fold=1
    # for train_idx,val_idx in splitter.split(train_names):
    #
    #   print ('training {} model'.format(fold))
    #   model.load_state_dict(torch.load('base.pkl'))
    #   train_set=[train_names[i] for i in train_idx]
    #   train_set = utils.get_train(train_set)
    #   val_set=[train_names[i] for i in val_idx]
    #
    #   train_datasest=ProteinDataset(dirpath=utils.TRAIN,fnames=train_set)
    #   train_loader = DataLoader(train_datasest, batch_size=batch_size, shuffle=True,num_workers=4)
    #
    #   val_dataset = ProteinDataset(dirpath=utils.TRAIN,fnames=val_set)
    #   val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #
    #
    #   train(model, options.epochs, train_loader,val_loader,str(fold) + '.pkl')
    #   fold+=1


if __name__ == "__main__":
    main()
