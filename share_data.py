import numpy as np
def get_shared_data(train_x,train_y,proportion):
  z = []
  for digit in range(10):
    all_samples = [i for i, d in enumerate(train_y) if d == digit]
    leng=len(all_samples)
    np.random.seed(660) #固定random结果
    random_share_index = np.random.choice(range(leng),int(leng*proportion))

    for i in random_share_index:
        z.append(all_samples[i])
  
  return train_x[z],train_y[z]
  


