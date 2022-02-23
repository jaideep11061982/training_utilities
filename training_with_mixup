
class AutoSave:
  def __init__(self, top_k=2, metric="f1", mode="min", root=None, name="ckpt"):
    self.top_k = top_k
    self.logs = []
    self.metric = metric
    self.mode = mode
    self.root = Path(root or MODEL_ROOT)
    assert self.root.exists()
    self.name = name

    self.top_models = []
    self.top_metrics = []

  def log(self, model, metrics):
    metric = metrics[self.metric]
    rank = self.rank(metric)

    self.top_metrics.insert(rank+1, metric)
    if len(self.top_metrics) > self.top_k:
      self.top_metrics.pop(0)

    self.logs.append(metrics)
    self.save(model, metric, rank, metrics["epoch"])


  def save(self, model, metric, rank, epoch):
    t = time.strftime("%Y%m%d%H%M%S")
    name = "{}_epoch_{:02d}_{}_{:.04f}".format(self.name, epoch, self.metric, metric)
    name = re.sub(r"[^\w_-]", "", name) + ".pth"
    path = self.root.joinpath(name)

    old_model = None
    self.top_models.insert(rank+1, name)
    if len(self.top_models) > self.top_k:
      old_model = self.root.joinpath(self.top_models[0])
      self.top_models.pop(0)      

    torch.save(model.state_dict(), path.as_posix())

    if old_model is not None:
      old_model.unlink()

    self.to_json()


  def rank(self, val):
    r = -1
    for top_val in self.top_metrics:
      if val <= top_val:
        return r
      r += 1

    return r
  
  def to_json(self):
    # t = time.strftime("%Y%m%d%H%M%S")
    name = "{}_logs".format(self.name)
    name = re.sub(r"[^\w_-]", "", name) + ".json"
    path = self.root.joinpath(name)

    with path.open("w") as f:
      json.dump(self.logs, f, indent=2)


def get_model(name, num_classes=NUM_CLASSES):
    """
    Loads a pretrained model. 
    Supports ResNest, ResNext-wsl, EfficientNet, ResNext and ResNet.

    Arguments:
        name {str} -- Name of the model to load

    Keyword Arguments:
        num_classes {int} -- Number of classes to use (default: {1})

    Returns:
        torch model -- Pretrained model
    """
    print(name)
    if "resnest3" in name:
        model = getattr(resnest_torch, name)(pretrained=True)
    elif "wsl" in name:
        model = torch.hub.load("facebookresearch/WSL-Images", name)
    elif name.startswith("resnext") or  name.startswith("resnet"):
        model = torch.hub.load("pytorch/vision:v0.6.0", name, pretrained=True)
    elif name.startswith("tf_efficientnet") :
        model = getattr(timm.models.efficientnet, name)(pretrained=True)
        print('name') 
    elif name.startswith("tf_mobile") :
        model = getattr(timm.models.mobilenetv3, name)(pretrained=True)
        print('name') 
    elif "efficientnet-b" in name:
        model = EfficientNet.from_pretrained(name)
    elif "resnes" in name:
        print(name)
        model = timm.create_model(name, pretrained=True,drop_rate =0.2)
    else:
        model = pretrainedmodels.__dict__[name](pretrained='imagenet')

    if hasattr(model, "fc"):
        nb_ft = model.fc.in_features
        model.fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "_fc"):
        nb_ft = model._fc.in_features
        model._fc = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "classifier"):
        nb_ft = model.classifier.in_features
        model.classifier = nn.Linear(nb_ft, num_classes)
    elif hasattr(model, "last_linear"):
        nb_ft = model.last_linear.in_features
        model.last_linear = nn.Linear(nb_ft, num_classes)

    return model
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_data_list(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x[0].size()[0]).to(DEVICE)

    mixed_x = [lam * i + (1 - lam) * i[index, :] for  i in x]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam



def one_step( xb, yb, net, criterion, optimizer, scheduler=None):
  #xb, yb = xb.to(DEVICE), yb.to(DEVICE)
  yb=yb.to(DEVICE)
  #xb, yb = [xb[0].to(DEVICE),xb[1].to(DEVICE),xb[2].to(DEVICE),xb[3].to(DEVICE)], yb.to(DEVICE)
  #xb, yb = [xb[0].to(DEVICE),xb[1].to(DEVICE) ], yb.to(DEVICE)
  if grp_length==1:

    xb = xb.to(DEVICE)
  else:

    xb = [xb[0].to(DEVICE),xb[1].to(DEVICE)] 

  if np.random.rand() < 0.0:
    if grp_length==1:

      xb, y_a, y_b, _ = mixup_data(xb.to(DEVICE), yb.to(DEVICE), alpha=5) #mixup_data_list
    else:
      xb, y_a, y_b, _ = mixup_data_list(xb , yb , alpha=5)  
    yb = torch.clamp(y_a + y_b, 0, 1)
        
  optimizer.zero_grad()
  o = net(xb)
 #loss = criterion(o, yb)
  loss=criterion(o,yb)
  loss.backward()
  optimizer.step()
  
  with torch.no_grad():
      l = loss.item()

      o = o.sigmoid()
      yb = (yb > 0.5 )*1.0
      lrap = label_ranking_average_precision_score(yb.cpu().numpy(), o.cpu().numpy())

      o = (o > 0.5)*1.0

      prec = (o*yb).sum()/(1e-6 + o.sum())
      rec = (o*yb).sum()/(1e-6 + yb.sum())
      f1 = 2*prec*rec/(1e-6+prec+rec)

  if  scheduler is not None:
    scheduler.step()

  return l, lrap, f1.item(), rec.item(), prec.item()


@torch.no_grad()
def evaluate(net, criterion, val_laoder):
    net.eval()

    os, y = [], []
    val_laoder = tqdm(val_laoder, leave = False, total=len(val_laoder))

    for icount, (xb, yb,f) in  enumerate(val_laoder):

        y.append(yb.to(DEVICE))
        if grp_length==1:

          xb = xb.to(DEVICE)
        else:

          xb = [xb[0].to(DEVICE),xb[1].to(DEVICE)] 
        #xb, yb = [xb[0].to(DEVICE),xb[1].to(DEVICE),xb[2].to(DEVICE),xb[3].to(DEVICE)] 
        #xb  = [xb[0].to(DEVICE),xb[1].to(DEVICE) ] 
        o = net(xb)

        os.append(o)

    y = torch.cat(y)
    o = torch.cat(os)

    l = criterion(o, y).item()
    
    o = o.sigmoid()
    y = (y > 0.5)*1.0

    lrap = label_ranking_average_precision_score(y.cpu().numpy(), o.cpu().numpy())

    o = (o > 0.5)*1.0

    prec = ((o*y).sum()/(1e-6 + o.sum())).item()
    rec = ((o*y).sum()/(1e-6 + y.sum())).item()
    f1 = 2*prec*rec/(1e-6+prec+rec)

    return l, lrap, f1, rec, prec, 
    
def one_epoch(net, criterion, optimizer, scheduler, train_laoder, val_laoder):
  net.train()
  l, lrap, prec, rec, f1, icount = 0.,0.,0.,0., 0., 0
  train_laoder = tqdm(train_laoder, leave = False)
  epoch_bar = train_laoder
  
  for (xb, yb,f) in  epoch_bar:
      #print(xb.size())
      # epoch_bar.set_description("----|----|----|----|---->")
      _l, _lrap, _f1, _rec, _prec = one_step(xb, yb, net, criterion, optimizer)
      l += _l
      lrap += _lrap
      f1 += _f1
      rec += _rec
      prec += _prec

      icount += 1
        
      if hasattr(epoch_bar, "set_postfix") and not icount%10:
          epoch_bar.set_postfix(
            loss="{:.6f}".format(l/icount),
            lrap="{:.3f}".format(lrap/icount),
            prec="{:.3f}".format(prec/icount),
            rec="{:.3f}".format(rec/icount),
            f1="{:.3f}".format(f1/icount),
          )
  
  scheduler.step()

  l /= icount
  lrap /= icount
  f1 /= icount
  rec /= icount
  prec /= icount
  
  l_val, lrap_val, f1_val, rec_val, prec_val = evaluate(net, criterion, val_laoder)
  
  return (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val)

TRAIN_BATCH_SIZE=144
VAL_BATCH_SIZE=144
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
steps = 303
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, T_mult=2, 
                                                           eta_min=1e-5, last_epoch=-1, verbose=False)
#scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
lr=[]
for epoch in range(50):
    for idx in range(steps):
        scheduler.step()
        #print(scheduler.get_lr())
        lr.append(scheduler.get_last_lr())
    
    #print('Reset scheduler')
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
   
def one_fold(model_name, fold, train_set, val_set, epochs=20, save=True, save_root=None):

    save_root = Path(save_root) or MODEL_ROOT

    saver = AutoSave(root=save_root, name=f"birdclef_{model_name}_fold{fold}", metric="f1_val")

  #net = get_model(model_name).to(DEVICE)
    #net = custom_model().to(DEVICE)
    #net.load_state_dict(torch.load('../input/clean-fast-simple-bird-identifier-training-colab/b1_concat_samp_0.6301685733057184_0_9'
    #                              ,map_location=DEVICE))

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(net.parameters(), lr=1e-3,weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, T_mult=2, 
                                                           eta_min=1e-5, last_epoch=-1, verbose=False)
    #optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=epochs)
    
  #train_data = BirdClefDataset(audio_image_store, meta=df.iloc[train_set].reset_index(drop=True),
  #                         sr=SR, duration=DURATION, is_train=True)
    print(len(df.iloc[train_set]))
    train_data = BirdClefDataset(audio_image_store,  meta=df.iloc[train_set].reset_index(drop=True), sr=SR, duration=DURATION, is_train=True)
    train_laoder = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, shuffle=True, pin_memory=True)
    #scheduler=  optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, 
    #                                                 T_max=len(train_data)//TRAIN_BATCH_SIZE)
  #val_data = BirdClefDataset(audio_image_store, meta=df.iloc[val_set].reset_index(drop=True),  sr=SR, duration=DURATION, is_train=False)
    val_data = BirdClefDataset(audio_image_store,  meta=df.iloc[val_set].reset_index(drop=True),  
                             sr=SR, duration=DURATION, is_train=False)
  
    val_laoder = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, num_workers=VAL_NUM_WORKERS, shuffle=False)
    print(len(train_data)//TRAIN_BATCH_SIZE)

    epochs_bar = tqdm(list(range(epochs)), leave=False)
    l_rap_best=-1
    for epoch  in epochs_bar:
        epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
        net.train()

        (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val) = one_epoch(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_laoder=train_laoder,
            val_laoder=val_laoder,
          )


        epochs_bar.set_postfix(
        loss="({:.6f}, {:.6f})".format(l, l_val),
        prec="({:.3f}, {:.3f})".format(prec, prec_val),
        rec="({:.3f}, {:.3f})".format(rec, rec_val),
        f1="({:.3f}, {:.3f})".format(f1, f1_val),
        lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
        )

        print(
            "[{epoch:02d}] loss: {loss} lrap: {lrap} f1: {f1} rec: {rec} prec: {prec}".format(
                epoch=epoch,
                loss="({:.6f}, {:.6f})".format(l, l_val),
                prec="({:.3f}, {:.3f})".format(prec, prec_val),
                rec="({:.3f}, {:.3f})".format(rec, rec_val),
                f1="({:.3f}, {:.3f})".format(f1, f1_val),
                lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
            )
        )

        if f1_val>l_rap_best:
            l_rap_best=f1_val
            print(f'saving model {fold} f1 {l_rap_best},epoch {epoch}   ')
            #torch.save(net.state_dict(),f'b2_concat_samp_{fold}')

        if save:
            metrics = {
              "loss": l, "lrap": lrap, "f1": f1, "rec": rec, "prec": prec,
              "loss_val": l_val, "lrap_val": lrap_val, "f1_val": f1_val, "rec_val": rec_val, "prec_val": prec_val,
              "epoch": epoch,
          }

            saver.log(net, metrics)
def train(model_name, epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix="", folds=None):
  gc.collect()
  torch.cuda.empty_cache()

  save_root = save_root or MODEL_ROOT/f"{model_name}{suffix}"
  save_root.mkdir(exist_ok=True, parents=True)
  
  fold_bar = tqdm(df.reset_index().groupby("fold").index.apply(list).items(), total=df.fold.max()+1)
  
  for fold, val_set in fold_bar:
      if folds and not fold in folds:
        continue
      
      print(f"\n############################### [FOLD {fold}]")
      fold_bar.set_description(f"[FOLD {fold}]")
      train_set = np.setdiff1d(df.index, val_set)
        
      one_fold(model_name, fold=fold, train_set=train_set , val_set=val_set , epochs=epochs, save=save, save_root=save_root)
    
      gc.collect()
      torch.cuda.empty_cache()   
      
      
MODEL_NAMES = [
      "effnetv2_s",
] 

MODEL_NAMES=['resnest26d']


for model_name in MODEL_NAMES:
  print("\n\n###########################################", model_name.upper())
  try:
    train(model_name, epochs=50, suffix=f"_sr{SR}_d{DURATION}_v1_v1", folds=[0])
    None
  except Exception as e:
    # print(f"Error {model_name} : \n{e}")
    raise ValueError() from  e
    

