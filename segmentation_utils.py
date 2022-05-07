#functions to convert encoding to mask and mask to encoding
def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T
  
def area_threshold(img,min_area=200,top_threshold=0.45):
    classification=img >top_threshold
    #print(classification.shape,img.shape)
    cont , heir = cv2.findContours(classification.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    areas = []
    for c in cont:
        areas.append(cv2.contourArea(c))
    
    max_area = np.max(areas)
    print(np.max(areas), np.min(areas), 'max',np.percentile(areas, 0.8), 'min',np.percentile(areas, 0.1))
    ''''
    if np.min(areas) > 600:
        min_area = np.percentile(areas, 0.1) 
    else:
        min_area = 500
    ''' 
    
    if np.min(areas) > 500:
        min_area = np.percentile(areas, 0.1) 
    else:
        min_area =  500
    
    
    
#     min_area = min(500, np.percentile(areas, 0.8))
#     min_area = 500 
#     print(min_area)
    
    for c in cont:
        area=cv2.contourArea(c)
        if area<min_area:
            zero_mask= np.zeros(img.shape,np.uint8)
#             cv2.drawContours(zero_mask,[c],0,255,-1)
            cv2.drawContours(zero_mask, [c], 0, 255, -1)
            
            c0,c1=np.nonzero(zero_mask)
            img[c0,c1]=0
            
    return img
  
  
def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs
def rle_encode_less_memory(img):
    #the image should be transposed
    pixels = img.T.flatten()
    
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    return ' '.join(str(x) for x in runs)
  
  
  def get_emptiness(arr):
    total_ele = arr.size
    mask=(arr<200 ) & (arr>2)
    
    #white_ele = np.count_nonzero(arr >= 200) + np.count_nonzero(arr <2)
    #print(white_ele/total_ele)
    return mask.sum() / total_ele
  
 #biuld slices from big tiff images
 class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce, encs=None):
        self.data = rasterio.open(os.path.join(DATA,idx+'.tiff'),num_threads='all_cpus')
        # some images have issues with their format 
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        
        
  
       
        self.reduce = reduce
        self.sz =  reduce*sz
        self.pad0 = (self.sz - self.shape[0]%self.sz)%self.sz
        self.pad1 = (self.sz - self.shape[1]%self.sz)%self.sz
        self.n0max = (self.shape[0] + self.pad0)//self.sz
        self.n1max = (self.shape[1] + self.pad1)//self.sz
        print(self.shape,self.n0max,self.n1max,self.sz)
        self.mask = enc2mask(encs,(self.shape[1],self.shape[0])) if encs is not None else None
        
    def __len__(self):
        return self.n0max*self.n1max
    
    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding (like in the previous version of the kernel)
        # then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0,n1 = idx//self.n1max, idx%self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0,y0 = -self.pad0//2 + n0*self.sz, -self.pad1//2 + n1*self.sz

        # make sure that the region to read is within the image
        p00,p01 = max(0,x0), min(x0+self.sz,self.shape[0])
        p10,p11 = max(0,y0), min(y0+self.sz,self.shape[1])
        img = np.zeros((self.sz,self.sz,3),np.uint8)
        mask = np.zeros((self.sz,self.sz),np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = np.moveaxis(self.data.read([1,2,3],
                window=Window.from_slices((p00,p01),(p10,p11))), 0, -1)
        else:
            for i,layer in enumerate(self.layers):
                img[(p00-x0):(p01-x0),(p10-y0):(p11-y0),i] =\
                  layer.read(1,window=Window.from_slices((p00,p01),(p10,p11)))
        if self.mask is not None: mask[(p00-x0):(p01-x0),(p10-y0):(p11-y0)] = self.mask[p00:p01,p10:p11]
        
        if self.reduce != 1:
            img = cv2.resize(img,(self.sz//reduce,self.sz//reduce),
                             interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask,(self.sz//reduce,self.sz//reduce),
                             interpolation = cv2.INTER_NEAREST)
            
            #img = cv2.resize(img,(1024,1024),
            #                 interpolation = cv2.INTER_AREA)
            #mask = cv2.resize(mask,(1024,1024),
            #                 interpolation = cv2.INTER_NEAREST)
        #check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)
        #return -1 for empty images
        #empti_chk=get_emptiness(img)
        return img, mask, (-1 if (s>s_th).sum() <= p_th or img.sum() <= p_th else idx)
x_tot,x2_tot = [],[]
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out,\
 zipfile.ZipFile(OUT_MASKS, 'w') as mask_out:
    for index, encs in tqdm(df_masks.iterrows(),total=len(df_masks)):
        #image+mask dataset
        ds = HuBMAPDataset(index,encs=encs)
        cnt=0
        for i in range(len(ds)):
            im,m,idx = ds[i]
            if idx < 0: continue
                
            x_tot.append((im/255.0).reshape(-1,3).mean(0))
            x2_tot.append(((im/255.0)**2).reshape(-1,3).mean(0))
            
            #write data  
            #im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.imwrite( f'{index}_{idx:04d}.png',im )
            img_out.write(f'{index}_{idx:04d}.png' )
            os.remove('/kaggle/working/' +  f'{index}_{idx:04d}.png'  )
             
            cv2.imwrite( f'{index}_{idx:04d}.png' ,m)
            mask_out.write(f'{index}_{idx:04d}.png')
            os.remove('/kaggle/working/' +  f'{index}_{idx:04d}.png'  )
            cnt=cnt+1
        print(f'file count for {index} {cnt} total ds count {len(ds)}')
            #im = cv2.imencode('.png',cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1]
            #img_out.writestr(f'{index}_{idx:04d}.png', im)
            #m = cv2.imencode('.png',m)[1]
            #mask_out.writestr(f'{index}_{idx:04d}.png', m)
        #break
#image stats
img_avr =  np.array(x_tot).mean(0)
img_std =  np.sqrt(np.array(x2_tot).mean(0) - img_avr**2)
print('mean:',img_avr, ', std:', img_std)

#sample view
columns, rows = 4,4
idx0 = 20
fig=plt.figure(figsize=(columns*4, rows*4))
with zipfile.ZipFile(OUT_TRAIN, 'r') as img_arch, \
     zipfile.ZipFile(OUT_MASKS, 'r') as msk_arch:
    fnames = sorted(img_arch.namelist())[8:]
    
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            print(fnames[idx0+idx])
            img = cv2.imdecode(np.frombuffer(img_arch.read(fnames[idx0+idx]), 
                                             np.uint8), cv2.IMREAD_COLOR)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = cv2.imdecode(np.frombuffer(msk_arch.read(fnames[idx0+idx]), 
                                              np.uint8), cv2.IMREAD_GRAYSCALE)
    
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow(Image.fromarray(img))
            plt.imshow(Image.fromarray(mask), alpha=0.2)
plt.show()

#models
      
df_masks = pd.read_csv(MASKS).set_index('id')
df_masks.head()


model = smp.DeepLabV3Plus(encoder_name=name, 
                         encoder_weights=cfg.encoder_weights, 
                         in_channels=cfg.in_channels, 
                         classes=cfg.classes)

