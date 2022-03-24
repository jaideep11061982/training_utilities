from multiprocessing import  Pool
from joblib import Parallel, delayed
def add_features(df):
    #df['question_text'] = df['question_text'].apply(lambda x:str(x))
    vocab['new_words']=vocab.raw_words.progress_apply(lambda x:spell(x) )
    
def parallelize_dataframe(df, func, n_cores=6):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

train = parallelize_dataframe(vocab, add_features)

def imgresize(path):
    img=cv2.imread(path)
    img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    img.save('/path',img)
    
Parallel(n_jobs=16,backend='multiprocessing')(delayed(imgresize)(path) for path in tqdm(paths))
