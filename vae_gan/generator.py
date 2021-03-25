# %% 
from operator import truediv
import numpy 
import os 
from torch.utils.data import Dataset, DataLoader, dataset 
import cv2
from skimage import filters, transform

numpy.random.seed(5)

def _resize(img):
    rescale_size = 64
    bbox = (40, 218 -30, 15, 178 - 15)
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # smooth image before resize to avoid moire patterns 
    scale = img.shape[0] / float(rescale_size)
    sigma = numpy.sqrt(scale) / 2.0
    img = filters.gaussian(img, sigma=sigma, multichannel=True)
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3, mode="constant")
    img = (img*255).astype(numpy.uint8)
    return img

# %%

class CELEBA(Dataset):

    def __init__(self, data_folder) -> None:
        super().__init__()
        # len is the number of files 
        self.len = len(os.listdir(data_folder))
        # list of file names 
        self.data_names = [os.path.join(data_folder, name) for name in sorted(os.listdir(data_folder))]

        self.len = len(self.data_names)

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, index):
        '''
        [summary]

        Args:
            index ([type]): image index between 0-(len-1)

        Returns:
            data.copy(): (data.copy(), data.copy()) image
        '''        
        # load image, crop 128x128, resize, transpose(to channel first), scale(so we can ust tanh)
        data = cv2.cvtColor(cv2.imread(self.data_names[index]), cv2.COLOR_BGR2RGB)

        data = _resize(data)

        # channel first 
        data = data.transpose(2, 0, 1)
        # tanh
        data = data.astype("float32") / 127.5 - 1.0

        return (data.copy(), data.copy())

class CELEBA_SLURM(Dataset):

    def __init__(self, data_folder):
        super().__init__()
        #open the file 
        self.file = open(data_folder, "rb")
        #get len 
        self.len = int(os.path.getsize(data_folder)) / (64*64*3)
    
    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, index):
        '''
        return the image 

        Args:
            index ([type]): image index between 0-(len-1)

        Returns:
            tensor: image 
        '''
        offset = index*3*64*64
        self.file.seek(offset) # 移动文件读取指针到指定位置
        data = numpy.fromfile(self.file, dtype=numpy.uint8, count=(3 * 64 * 64))
        data = numpy.reshape(data, newshape=(3, 64, 64))
        data = data.astype("float32") / 127.5 - 1.0
        return (data.copy(), data.copy())

# %%

if __name__ == "__main__":
    # 这里是数据集的位置
    dataset = CELEBA_SLURM(".")
    # dataset = CELEBA(r"H:/data/img_align_celeba/")
    gen = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
    from matplotlib import pyplot
    imgs = []
    for i, (b, l) in enumerate(gen):
        print("{}:{}".format(i, len(gen)))

    # for i in range(1000):

    #     a = gen.__iter__().next()
    #     # scale between (0, 1)
    #     a = (a + 1) / 2
    #     for el in a:
    #         pyplot.imshow(numpy.transpose(el.numpy(), (1, 2, 0)))
    #         pyplot.show()
 
# %%
