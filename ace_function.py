# ACE detector function
# DataSet = Hyp.Image resized in [npixels,nbands]
# tgtSet  = Set of target signatures [ntgts,nbands]

import numpy as np 
from numpy.linalg import inv
import math
import spectral.io.envi as envi
from spectral import *
from io import StringIO
import scipy.io

def ACE_rule(DataSet,tgtSet):
# ------ Get size of DataSet and Target(s)-----
    DataSetTr   = DataSet.transpose()     #[nbands,npixels]
    DataSize = np.shape(DataSetTr)
    nbands = DataSize[0]
    npixels = DataSize[1]
    TgtSize = np.shape(tgtSet.transpose()) #[nbands,1]
    ntgts = TgtSize[1] 
# ------ Mean-center by removing data mean from data and target(s).-----
    u = np.sum(DataSetTr, axis = 1)/npixels        #[nbands,1]
    u = u.reshape((nbands,1))
    DataSetCtr = DataSetTr - (u@np.ones((1,npixels))) #[nbands,npixels]
    TgtSetCtr  = tgtSet.transpose() - (u@np.ones((1,ntgts)))     #[nbands,1]
# ------ Compute covariance matrix of mean-centered data ------
    covDataSetCtr = (DataSetCtr@DataSetCtr.transpose())/(npixels-1)  #[nbands,nbands]
#    covDataSetCtr = (DataSet.transpose()@DataSet)/(npixels-1);
    covDataInv = inv(covDataSetCtr)                       #[nbands,nbands]
# ------ Compute ACE score for each pixel ---------
    ACE_image = np.zeros((npixels,1))
    tgt = TgtSetCtr                     #[nbands,1]
#tgt  = tgtSet;
    temp = math.sqrt(tgt.transpose()@covDataInv@tgt)   #sqrt is added here and in ACE_rule
    for i in range(1,npixels):
        x = DataSetCtr[:,i]           #[nbands,1]
        ACE_image[i,:] = nbands*(tgt.transpose()@covDataInv@x)/(temp*math.sqrt(x.transpose()@covDataInv@x))  
    return ACE_image

#img = envi.open('001_0729-1919_QUAC_refl_146x84_chip-ReCalibrated.img.hdr','001_0729-1919_QUAC_refl_146x84_chip-ReCalibrated.img')
#rgbimg = img[[54,34,14]]
##view = imshow(img, bands=(54, 34, 14))
#rows = img.shape[0]
#cols = img.shape[1]
#txtdata = np.loadtxt('BlueTFieldGrassShare2012_ClnSampd.txt')
#mat = scipy.io.loadmat('bbl_SHARE2010.mat')
#true_bands = np.argwhere(mat['bbl'] == 1)
##print(true_bands[:,1])
#Image = img.read_bands(true_bands[:,1])
#bands = Image.shape[2]
#tgt = txtdata[true_bands[:,1],1]
#tgtsig = tgt.reshape((1,bands))
##print(bands)
#DataSet = np.reshape(Image,[rows*cols,bands])
#ACE  = ACE_rule(DataSet,tgtsig)
#ACE_image = ACE.reshape((rows,cols))
##ACE_image = ace(Image,tgtsig)
#print(ACE_image.shape)
##view = imshow(ACE_image,stretch=True)

