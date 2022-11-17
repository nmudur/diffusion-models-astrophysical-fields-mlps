import os
import torch
import torchvision
import numpy as np
from numpy.fft import *
#from torchvision.transforms import Compose, 
from torchvision.utils import save_image

import torch.nn.functional as F
import torchvision.transforms as T
from tqdm.auto import tqdm
from torch import nn


import hf_diffusion

def get_fieldidx_for_index(idx):
    return np.arange(idx*15, (idx+1)*15, dtype=int)

def preprocess_resize_field(path, Nx):
    field = np.load(path)
    tfields = torch.tensor(np.log10(field)) #take log field
    resizedfields = T.Resize([Nx, Nx], interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True)(tfields)
    return resizedfields


def save_fields_as_png(ftensor, directory, abbrev):
    if os.path.isdir(directory):
        print("Error: directory already exists")
    else:
        os.mkdir(directory)
    for i in range(ftensor.shape[0]):
        save_image(ftensor[i], os.path.join(directory, abbrev+str(i).zfill(5)+'.jpeg'))
    return


def augment_fields(fields, transformation_list):
    #fields: numpy array
    assert len(fields.shape)==4 #BCHW
    tfields = torch.tensor(fields)
    tflist = [tfields]
    for transform in transformation_list:
        tflist.append(transform(tfields))
    return torch.cat(tflist).numpy()


def calc_1dps_img2d(kvals, img, to_plot=True, smoothed=0.5):
    Nx = img.shape[0]
    fft_zerocenter = fftshift(fft2(img)/Nx**2) #Aug
    impf = abs(fft_zerocenter) ** 2.0
    x, y = np.meshgrid(np.arange(Nx), np.arange(Nx))
    R  = np.sqrt((x-(Nx/2))**2+(y-(Nx/2))**2) #Aug
    filt = lambda r: impf[(R >= r - smoothed) & (R < r + smoothed)].mean()
    mean = np.vectorize(filt)(kvals)
    return mean


def get_samples_given_saved_dict(sdpath, numsamples, samplabels=None, device='cpu'):
    sdict = torch.load(sdpath, map_location='cpu')
    if sdict['model_type']=='baseline':
        if 'dim' not in sdict['model_kwargs'].keys():
            mkw = sdict['model_kwargs']
            mkw['dim'] = mkw.pop('image_size')
            model = hf_diffusion.Unet(**mkw)
        else:
            model = hf_diffusion.Unet(**sdict['model_kwargs'])
    else:
        raise NotImplementedError()

    model.load_state_dict(sdict['model_state_dict'])
    model.to(device)
    betas = sdict['betas']
    diff = hf_diffusion.Diffusion(betas)
    print('Beta shape', betas.shape)
    samples = diff.sample(model, sdict['model_kwargs']['dim'], numsamples, sdict['model_kwargs']['channels'], samplabels)
    return samples


def denoise_images(noisy_images, trsigma, sdict, transformations=[nn.Identity(), nn.Identity()]):
    '''
    :param noisy_image: Image+Noise B*C*Nx*Nx
    :param sigma: Noises added to Image in the transformed space
        trsigma = sigma_data * 2 / (RANGE_MAX - RANGE_MIN)

    :param sdict: Saved checkpoint to use model
    :param transformations: whether to transform before and after,
    :return:
    '''
    B = noisy_images.shape[0]
    tr, invtr = transformations
    trnoisyimages = tr(torch.tensor(noisy_images, dtype=torch.float32))

    #load model
    sdict = torch.load(sdict, map_location='cpu')
    diff = hf_diffusion.Diffusion(sdict['betas'])
    model = hf_diffusion.Unet(**sdict['model_kwargs'])
    model.load_state_dict(sdict['model_state_dict'])

    #find what timestep the noise levels map to
    tn_index = np.zeros(B, dtype=int)
    for b in range(B):
        mindiff = np.abs(diff.sqrt_one_minus_alphas_cumprod.numpy() - trsigma[b])
        tn_index[b] = np.where(mindiff == np.min(mindiff))[0]
    print('Timesteps corr to noise', tn_index)

    #need a loop since different numbers of steps :(
    denoised_images = []
    invtr_denoised_images = []
    for b in range(B):
        trnoisyimg = torch.unsqueeze(trnoisyimages[b], 0)
        t_input = torch.tensor(tn_index[b])
        t_reversed = np.flip(np.arange(0, int(t_input.numpy())))
        imgs_denoising = []
        img_tminus1 = trnoisyimg
        for t in t_reversed:
            img_tminus1 = diff.p_sample(model, img_tminus1, t=torch.tensor(np.array([t])), t_index=t)
            imgs_denoising.append(img_tminus1.numpy()[0, 0, ...]) #images T-1 to 0 given the noisy image
        denoised_images.append(imgs_denoising[-1])
        invtr_denoised_images.append(invtr(torch.unsqueeze(torch.unsqueeze(torch.tensor(imgs_denoising[-1]), dim=0), dim=0))[0, 0].numpy())

    return denoised_images, invtr_denoised_images
