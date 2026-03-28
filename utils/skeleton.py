# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import random

try:
    from utils import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,

                 l_ratio,
                 input_size,
                 input_representations,
                 mmap=True):

        self.data_path = data_path
        # self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representations=input_representations
        self.crop_resize =True
        self.l_ratio = l_ratio


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print(self.data.shape)
        # print(self.data.shape,len(self.number_of_frames))
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
       ### # self.number_of_frames= np.load(self.num_frame_path)



    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        ##### number_of_frames = self.number_of_frames[index]

        # apply spatio-temporal augmentations to generate  view 1 

        # temporal crop-resize
        ### data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations 

        # flip_prob  = random.random()
        # if flip_prob < 0.5:
        #          data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1_crop)
        # else:
        #          data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)

        # ---------------------------------------------
        # flip_prob  = random.random()
        # if flip_prob < 0.5:
        #          data_numpy_v1 = augmentations.pose_augmentation(data_numpy)
        # else:
        #          data_numpy_v1 = augmentations.joint_courruption(data_numpy)

        data_numpy_v1 = augmentations.joint_courruption(data_numpy)


        # ---------------------------------------------





        # apply spatio-temporal augmentations to generate  view 2

        # temporal crop-resize
        ### data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations 


        # flip_prob  = random.random()
        # if flip_prob < 0.5:
        #          data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
        # else:
        #          data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        #---------------------------------------------
        # flip_prob = random.random()
        # if flip_prob < 0.5:
        #     data_numpy_v2 = augmentations.pose_augmentation(data_numpy)
        # else:
        #     data_numpy_v2 = augmentations.joint_courruption(data_numpy)
         # ---------------------------------------------


        # convert augmented views into input formats based on skeleton-representations

        if self.input_representations == "seq-based_and_graph-based" or self.input_representations == "seq-based_and_image-based"  :

             # Input View 1
             #sequence-based input of view 1 ---> shpae (64 X 150)
             input_s1_v1 = data_numpy_v1.transpose(1,2,0,3)
             input_s1_v1 = input_s1_v1.reshape(-1,self.C * self.V * self.M).astype('float32')
             #graph-based / image-based input of view 1 ---> shape (3, 64, 25, 2)
             input_s2_v1 = data_numpy_v1.astype('float32')

             # # Input View 2
             # #sequence-based input of view 2 ---> shpae (64 X 150)
             # input_s1_v2 = data_numpy_v2.transpose(1,2,0,3)
             # input_s1_v2 = input_s1_v2.reshape(-1,self.C * self.V * self.M).astype('float32')
             # #graph-based / image-based input of view 2 ---> shape (3, 64, 25, 2)
             # input_s2_v2 = data_numpy_v2.astype('float32')

        elif self.input_representations == "graph-based_and_image-based":  #不用这个

             # Input View 1
             #graph-based and image-based inputs of view 1 ---> shape (3, 64, 25, 2)
             input_s1_v1 = data_numpy_v1.astype('float32')
             input_s2_v1 = data_numpy_v1.astype('float32')

             # # Input View 2
             # #graph-based and image-based inputs of view 2 ---> shape (3, 64, 25, 2)
             # input_s1_v2 = data_numpy_v2.astype('float32')
             # input_s2_v2 = data_numpy_v2.astype('float32')

        return  input_s1_v1,input_s2_v1

        # return input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2


# def main():
#     S = Feeder('/public/yyx/pywork/skeleton-contrast-main/data/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy','/public/yyx/pywork/skeleton-contrast-main/data/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy'
#                ,[0.1, 1],64, 'seq-based_and_graph-based')
#
#     # S = Feeder('//home/user0/Public/yyx/pywork/dataset/cmu/train/basketball/basketball_1.npy',
#     #             [0.1, 1], 64, 'seq-based_and_graph-based')
#     a=[]
#     # for i in range(self.N):
#     #     a.append(S[i])
#     a, _, _, _ = S[1]
#     #i指的是N（batch=1)的次数迭代
#     print(a)
#     pass
# if __name__=="__main__":
#     main()


#'/public/yyx/pywork/skeleton-contrast-main/data/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy'

#'//home/user0/Public/yyx/pywork/dataset/cmu/train/basketball/basketball_1.npy'