#!/usr/bin/env python3
from ace_function import ACE_rule

def do_main():
    img = envi.open('001_0729-1919_QUAC_refl_146x84_chip-ReCalibrated.img.hdr','001_0729-1919_QUAC_refl_146x84_chip-ReCalibrated.img')
    rgbimg = img[[54,34,14]]
    #view = imshow(img, bands=(54, 34, 14))
    rows = img.shape[0]
    cols = img.shape[1]
    txtdata = np.loadtxt('BlueTFieldGrassShare2012_ClnSampd.txt')
    mat = scipy.io.loadmat('bbl_SHARE2010.mat')
    true_bands = np.argwhere(mat['bbl'] == 1)
    #print(true_bands[:,1])
    Image = img.read_bands(true_bands[:,1])
    bands = Image.shape[2]
    tgt = txtdata[true_bands[:,1],1]
    tgtsig = tgt.reshape((1,bands))
    #print(bands)
    DataSet = np.reshape(Image,[rows*cols,bands])
    ACE  = ACE_rule(DataSet,tgtsig)
    ACE_image = ACE.reshape((rows,cols))
    #ACE_image = ace(Image,tgtsig)
    print(ACE_image.shape)
    view = imshow(ACE_image,stretch=True)

do_main()