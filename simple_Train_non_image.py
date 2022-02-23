class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, w):
        if y is None:
            y = np.zeros(len(X), dtype=np.float32)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.w = w.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i]

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
        
 criterion = torch.nn.L1Loss()
criterion_val= torch.nn.L1Loss()
#VentilatorLoss()
def evaluate(model, loader_val):
    tb = time.time()
    was_training = model.training
    model.eval()

    loss_sum = 0
    score_sum = 0
    n_sum = 0
    y_pred_all = []
    losses_val=AverageMeter()
    for ibatch, (x, y, w) in enumerate(loader_val):
        n = y.size(0)
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        with torch.no_grad():
            y_pred = model(x).squeeze()

        loss = criterion(y_pred, y)
        #loss = criterion_val(y_pred, y,w).mean()
        losses_val.update(loss.item(),n)
        n_sum += n
        loss_sum += n*loss.item()
        if ibatch % 10 == 0 or ibatch == (len(loader_val)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      #'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      #'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.6f}  '
                      .format(
                       iepoch+1, ibatch, len(loader_val),
                       #remain=timeSince(start, float(step+1)/len(train_loader)),
                       loss=losses_val,
                       
                       lr=1e-3,
                       ))
        
        y_pred_all.append(y_pred.cpu().detach().numpy())

    loss_val = loss_sum / n_sum

    model.train(was_training)

    d = {'loss': loss_val,
         'time': time.time() - tb,
         'y_pred': np.concatenate(y_pred_all, axis=0)}

    return d
    
from tqdm.notebook import tqdm as tqdm
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR,ReduceLROnPlateau


nfold = 5
kfold = KFold(n_splits=nfold, shuffle=True, random_state=228)
epochs = 2 if debug else 292
lr = 1e-3
batch_size = 1024
max_grad_norm = 1000
import time
log = {}
oof_pred = []
oof_target = []
oof_ids=[]
for ifold, (idx_train, idx_val) in enumerate(kfold.split(X_all)):
    print('Fold %d' % ifold)
    tb = time.time()
    model = Model(input_size)
    model.to(device)
    model.train()
    if ifold > 2: # due to time limit
        break
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.55, patience=10,verbose=True)
 
    X_train = X_all[idx_train]
    y_train = y_all[idx_train]
    w_train = w_all[idx_train]
    ids_val=ids[idx_val]
    X_val = X_all[idx_val]
    y_val = y_all[idx_val]
    w_val = w_all[idx_val]

    dataset_train = Dataset(X_train, y_train, w_train)
    dataset_val = Dataset(X_val, y_val, w_val)
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=False,
                         batch_size=batch_size, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, shuffle=False,
                         batch_size=batch_size, drop_last=False)
    '''
    scheduler=OneCycleLR(optimizer, 2e-3, total_steps=len(loader_train)*batch_size
                                 ,cycle_momentum=False, 
                                 pct_start=0.6, anneal_strategy='cos',
                                 div_factor=30.0, final_div_factor=1,
                                 last_epoch=-1, verbose=False)
    '''
    scheduler = ReduceLROnPlateau(optimizer, factor=0.55, patience=10)
    
    losses_train = []
    losses_val = []
    lrs = []
    time_val = 0
    best_score = np.inf
   
    print('epoch loss_train loss_val lr time')
    best_score=np.inf
    for iepoch in range(epochs):
        loss_train = 0
        n_sum = 0
        losses = AverageMeter()
        for ibatch, (x, y, w) in enumerate(tqdm(loader_train)):
            n = y.size(0)
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            optimizer.zero_grad()

            y_pred = model(x).squeeze()
            loss = criterion_val(y_pred,y )
            #loss = criterion_val(y_pred, y,w).mean()
            losses.update(loss.item(),n)
            loss_train += n*loss.item()
            n_sum += n

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if ibatch % 100 == 0 or ibatch == (len(loader_train)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      #'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      #'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.6f}  '
                      .format(
                       iepoch+1, ibatch, len(loader_train),
                       #remain=timeSince(start, float(step+1)/len(train_loader)),
                       loss=losses,
                       
                       lr=1e-3,
                       #lr=scheduler.get_last_lr()[-1] 
                       ))

            optimizer.step()
            #scheduler.step()
        val = evaluate(model, loader_val)
        
        loss_val = val['loss']
        time_val += val['time']
        if val['loss']<best_score:
            best_score=val['loss']
            time.sleep(1.5)
            print('saving best wt',best_score)
            torch.save(model.state_dict(),f'fold_best{ifold}.pt')
        losses_train.append(loss_train / n_sum)
        losses_val.append(val['loss'])
        lrs.append(optimizer.param_groups[0]['lr'])

        print('%3d %9.6f %9.6f %7.3e %7.1f %6.1f' %
              (iepoch + 1,
               losses_train[-1], losses_val[-1], 
               lrs[-1], time.time() - tb, time_val))

        scheduler.step(losses_val[-1])
        #capure oof
        #oof_pred.append( val['y_pred'])
        #oof_target.append(y_val)
        #oof_ids.append(ids_val)
    print('generate fold',ifold ,' oofs')
    model.load_state_dict(torch.load(f'/kaggle/working/fold_best{ifold}.pt'))
    val = evaluate(model, loader_val)
    oof_pred.append(val['y_pred'])
    oof_target.append(y_val)
    oof_ids.append(ids_val)
    #oof_target=np.concatenate(oof_target)
    #oof_ids=np.concatenate(oof_ids)
    ofilename = 'model%d.pth' % ifold
    torch.save(model.state_dict(), ofilename)
    print(ofilename, 'written')

    log['fold%d' % ifold] = {
        'loss_train': np.array(losses_train),
        'loss_val': np.array(losses_val),
        'learning_rate': np.array(lrs),
        'y_pred': val['y_pred'],
        'idx': idx_val
        
        
 features = create_features(test)
features = rs.transform(features)

X_test = features.reshape(-1, 80, features.shape[-1])
y_test = np.zeros(len(features)).reshape(-1, 80)
w_test = 1 - test.u_out.values.reshape(-1, 80)

dataset_test = Dataset(X_test, y_test, w_test)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

y_pred_folds = np.zeros((len(test), 2), dtype=np.float32)
for ifold in range(2):
    model = Model(input_size)
    model.to(device)
    model.load_state_dict(torch.load('model%d.pth' % ifold, map_location=device))
    model.eval()
    
    y_preds = []
    for x, y, _ in loader_test:
        x = x.to(device)
        with torch.no_grad():
            y_pred = model(x).squeeze()

        y_preds.append(y_pred.cpu().numpy())
    
    y_preds = np.concatenate(y_preds, axis=0)
    y_pred_folds[:, ifold] = y_preds.flatten()

submit.pressure = np.mean(y_pred_folds, axis=1)
submit.to_csv('submission.csv', index=False)
print('submission.csv written')
    }
