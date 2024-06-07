import scipy as sp
import numpy as np
import scipy.ndimage as nd   


def LofG(img,sigma=1., thresh=0.1):
    LoG = nd.gaussian_laplace(img.float(),sigma)        # Calculate LoG image
    th = 10*np.absolute(LoG).mean()*thresh              # Calculate threshold
    output = sp.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y-1:y+2, x-1:x+2]               # Get 3x3 patch
            if patch[1,1]<0:                            # Check all cases of [-] in the middle
                if (patch.max()>0) and (patch.max()-patch[1,1]>th):
                    output[y, x] = 1;
            elif patch[1,1]==0:
                if (np.sign(patch[0,1])*np.sign(patch[2,1])<0) and (np.absolute(patch[2,1]-patch[0,1])>th): # Check [- 0 +]', [+ 0 -]'
                    output[y, x] = 1;
                if (np.sign(patch[1,0])*np.sign(patch[1,2])<0) and (np.absolute(patch[1,2]-patch[1,0])>th): # Check [- 0 +], [+ 0 -]
                    output[y, x] = 1;
                if (np.sign(patch[0,0])*np.sign(patch[2,2])<0) and (np.absolute(patch[2,2]-patch[0,0])>th): # Check main diagonal
                    output[y, x] = 1;
                if (np.sign(patch[2,0])*np.sign(patch[0,2])<0) and (np.absolute(patch[0,2]-patch[2,0])>th): # Check inverse diagonal
                    output[y, x] = 1;
    return output