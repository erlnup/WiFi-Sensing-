import glob 
import scipy.io as sio
import numpy as np


NTU_FI = '.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID'


"""
NTU_Fi_HumanID  test_amp  train_amp

NTU_Fi_HumanID:
 test_amp   train_amp


train_amp:

 001   002   003   004   005   006   007   008   009   010   011   012   013   015

test_amp:
 001   002   003   004   005   006   007   008   009   010   011   012   013   015

 001:
 a0.mat    a11.mat   a3.mat   a6.mat   a9.mat   b10.mat   b2.mat   b5.mat   b8.mat   c1.mat    c12.mat   c4.mat   c7.mat
 a1.mat    a12.mat   a4.mat   a7.mat   b0.mat   b11.mat   b3.mat   b6.mat   b9.mat   c10.mat   c2.mat    c5.mat   c8.mat
 a10.mat   a2.mat    a5.mat   a8.mat   b1.mat   b12.mat   b4.mat   b7.mat   c0.mat   c11.mat   c3.mat    c6.mat   c9.mat

"""

full_path = '/Users/erlnup/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a0.mat'


mat = sio.loadmat('/Users/erlnup/.cache/kagglehub/datasets/hylanj/wifi-csi-dataset-ntu-fi-humanid/versions/1/NTU-Fi-HumanID/test_amp/001/a0.mat')

t = mat['CSIamp']

print(f' Type {type(t)}')
print(f' Shape CSIamp: {np.shape(t)}')
print(f' First element {t[0]}')
print(f'First element Type {type(t[0])}')
print(f'test {t[0][0]}')




