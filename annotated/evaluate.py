import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from quantimpy import minkowski as mk

import hf_diffusion
import utils

@torch.no_grad()
def get_loss_for_samples(model, diffusion, loss_type, timesteps_eval, samples_list, numnoise, labels=None, averaged=False, device='cpu'):
    '''
    :param model: trained unet
    :param diffusion:
    :param loss_type: Loss
    :param timesteps_eval: Tiemsteps to evlauate the loss at
    :param samples_list: List of images, each elem is Nimgx1xHxW
    :param numnoise: num noise samples for EACH timestep
    :return: List of NimgxTxnumnoise i.e. the 'loss' at each timestep
    (how bad the prediction is
    for the noise added at each timestep)
    '''
    assert labels is None
    loss_list = []
    timebatch = torch.tensor(np.hstack([timesteps_eval] * numnoise), device=device)  # Numnoise*|TE|: te changes faster
 
    BATCHSIZE = len(timebatch)
    print(BATCHSIZE, timebatch.shape)
    for isa, sample in enumerate(samples_list):
        #'loss' should be T*Numnoise
        loss_inner = [] #for all images in a sample
        for imgidx in range(sample.shape[0]):
            img = sample[imgidx].view(1, 1, sample.shape[-2], sample.shape[-1])
            batch = torch.vstack([img]*BATCHSIZE)
            model.to(device)
            batch = batch.to(device)
            #print('ckp3', batch.shape)
            noise = torch.randn_like(batch, device=device) #(Numnoise*T) * 1 * H*W
            x_t = diffusion.q_sample(x_start=batch, t=timebatch, noise=noise)

            predicted_noise = model(x_t, timebatch, labels)
            if imgidx==0:
                print('loss', diffusion.p_losses(model, batch, timebatch, loss_type=loss_type).item()) #would return only 1 number averaged over the batch

            if loss_type == 'l1':
                loss = F.l1_loss(noise, predicted_noise, reduction='none')
            elif loss_type == 'l2':
                loss = F.mse_loss(noise, predicted_noise, reduction='none')
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise, reduction='none')
            else:
                raise NotImplementedError()

            loss = torch.mean(loss, dim=[-3, -2, -1])
            loss = loss.view((numnoise, len(timesteps_eval))).transpose(0, 1)
            loss_inner.append(loss)
        loss_samp = torch.stack(loss_inner) #Nimage*T*N
        item = loss_samp.cpu().numpy() if device=='cuda' else loss_samp.numpy()
        if averaged:
            loss_list.append(np.mean(item, axis=2)) #Nimage * T
        else:
            loss_list.append(item)
        print('__________')
    return loss_list



def plot_ps_samples(kvals, samplist, names, cols=['b', 'r'], logscale=True, k2pk=False, savefig_dict={}):
    '''
    :param kvals:
    :param samplist: List of power spectra for samples (eg: either from different models or the real fields)
    :param names:
    :param cols:
    :return:
    '''
    
    plt.figure(figsize=savefig_dict['figsize'] if 'figsize' in savefig_dict.keys() else [6, 6])
    for isd, samp in enumerate(samplist):
        assert len(samp.shape)==2
        if k2pk:
            samp = samp*(kvals**2) #check this line
        meanps = np.mean(samp, axis=0)
        stdps = np.std(samp, axis=0, ddof=1)
        style='solid' if isd==0 else 'dashed'
        plt.plot(kvals, meanps, c=cols[isd], label=names[isd], linestyle=style)
        plt.fill_between(kvals, meanps-stdps, meanps+stdps, alpha=0.2, color=cols[isd])
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel(r'k')
    if k2pk:
        plt.ylabel(r'$k^2P(k)$')
    else:
        plt.ylabel(r'P(k)')
    plt.legend()
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return



def get_powspec_for_samples(samplist):
    '''
    :param samplist: list of np arrays with shape N_img, Nx, Nx
    :param hist_kwargs: bins, range, density
    :return:
    '''
    ps_list = []
    Nx = samplist[0].shape[-1]
    kvals = np.arange(0, Nx/2)
    for samp in samplist:
        assert len(samp.shape)==3
        assert samp.shape[-1]==Nx
        assert samp.shape[-2]==Nx
        pssamp = np.vstack([utils.calc_1dps_img2d(kvals, samp[ci, ...], to_plot=False, smoothed=0.25) for ci in range(samp.shape[0])])
        ps_list.append(pssamp)
    return kvals, ps_list



def get_pixel_histogram_for_samples(samplist, hist_kwargs, names, cols, with_err=True, savefig_dict={}):
    '''
    :param samplist: list of torch tensors with shape N_img, Nx, Nx
    :param hist_kwargs: bins, range, density
    :return:
    '''
    sampwise_histmean  = []
    sampwise_histstd = []
    for samp in samplist:
        hist_all = np.zeros((samp.shape[0], len(hist_kwargs['bins'])-1))
        for img in range(samp.shape[0]):
            vals = np.histogram(samp[img][:], **hist_kwargs)
            hist_all[img, :] = vals[0]
        sampwise_histmean.append(hist_all.mean(0))
        sampwise_histstd.append(np.std(hist_all, axis=0, ddof=1))
    #bins
    bins = hist_kwargs['bins']
    bins_low = bins[:-1]
    bins_upp = bins[1:]
    bins_mid = (bins_upp+bins_low)/2
    bins_width = bins_upp - bins_low
    plt.figure()
    for isa in range(len(samplist)):
        if with_err:
            plt.bar(bins_mid, sampwise_histmean[isa], yerr=sampwise_histstd[isa]/np.sqrt(len(samplist[0])),width=bins_width,
                 label=names[isa], color=cols[isa], alpha=0.2, ecolor=cols[isa])
        else:
            plt.bar(bins_mid, sampwise_histmean[isa], width=bins_width,
                 label=names[isa], color=cols[isa], alpha=0.2)
    plt.legend()
    plt.xlabel('Pixel intensity')
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return sampwise_histmean, sampwise_histstd



def plot_mink_functionals(samplist, gs_vals, names, cols, savefig_dict={}):
    sampwise_minkmean  = []
    sampwise_minkstd = []
    for samp in samplist:
        samp_minks = []
        for isa in range(len(samp)):#each image
            image = samp[isa]
            gs_masks = [image>=gs_vals[ig] for ig in range(len(gs_vals))]
            minkowski = []
            for i in range(len(gs_masks)):
                minkowski.append(mk.functionals(gs_masks[i], norm=True))
            minkowski = np.vstack(minkowski) #N_alphax3
            samp_minks.append(minkowski)
        samp_minks = np.stack(samp_minks) #NsampxN_alphax3
        sampwise_minkmean.append(samp_minks.mean(0))
        sampwise_minkstd.append(np.std(samp_minks, axis=0, ddof=1))
    
    fig, ax = plt.subplots(figsize=(10, 15), nrows=3)
    for iax in range(3):
        for isa in range(len(samplist)):
            style='solid' if isa==0 else 'dashed'
            ax[iax].plot(gs_vals, sampwise_minkmean[isa][:, iax], cols[isa], label=names[isa], linestyle=style)
            ax[iax].fill_between(gs_vals, sampwise_minkmean[isa][:, iax]-sampwise_minkstd[isa][:, iax], 
                    sampwise_minkmean[isa][:, iax]+sampwise_minkstd[isa][:, iax], color=cols[isa], alpha=0.2)
        ax[iax].set_xlabel('g')
        if iax==0:
            ax[iax].set_ylabel(r'$\mathcal{M}_{0}(g)$', fontsize=18)
        elif iax==1:
            ax[iax].set_ylabel(r'$\mathcal{M}_{1}(g)$', fontsize=18)
        else:
            ax[iax].set_ylabel(r'$\mathcal{M}_{2}(g)$', fontsize=18)
        if iax==0:
            ax[iax].legend(prop={'size': 20})
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return sampwise_minkmean, sampwise_minkstd
            
def plot_panel_images(images,titles, nrow, ncol, figsize, savefig_dict):
    fig, ax = plt.subplots(figsize=figsize, nrows = nrow, ncols=ncol)
    ax = ax.ravel()
    vmin, vmax = savefig_dict['vmin'] if 'vmin' in savefig_dict.keys() else None, savefig_dict['vmax'] if 'vmax' in savefig_dict.keys() else None
    for ii, img in enumerate(images):
        c = ax[ii].imshow(img, origin='lower', vmin=vmin, vmax=vmax)
        if 'no_colorbar' not in savefig_dict.keys():
            plt.colorbar(c, ax=ax[ii], fraction=0.05)
        if titles is not None:
            ax[ii].set_title(titles[ii])
        ax[ii].axis('off')
    if 'wspace' in savefig_dict.keys():
        fig.subplots_adjust(wspace=savefig_dict['wspace'])
    if 'save_path' in savefig_dict.keys():
        plt.savefig(savefig_dict['save_path'], dpi=savefig_dict['dpi'] if 'dpi' in savefig_dict else 100, bbox_inches='tight')
    plt.show()
    return
        
        
    


    
if __name__=='__main__':
    resdir = 'results/samples_exps/'
    #Part 1: Looking at the losses of a poorly trained model: low loss, bad samples
    #the loss for hte earlier timesteps (nearer the data dbn) is MUCH worse for both the real images and the samples
    #the samples also had a lower loss for the earlier timesteps
    #i.e. if your schedule prioritized minimizing the loss of earlier steps that might be better
    #test losses
    #the loss for the sampled images is also higher

    '''
    run = 'Run_8-19_14-44'
    badfields = np.load(os.path.join(resdir, run, '/samples_100.npy'))
    fields64 = np.load('../../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx64_train.npy')

    numimgs = 10
    rng = np.random.default_rng(seed=23)
    selidx = rng.choice(100, numimgs, replace=False)

    #transform
    RMIN, RMAX = torch.tensor(np.min(fields64)), torch.tensor(np.max(fields64))
    tr, invtr = hf_diffusion.get_minmax_transform(RMIN, RMAX)

    realsamps = tr(torch.unsqueeze(torch.from_numpy(fields64[selidx, ...]), 1))
    badsamps = tr(torch.unsqueeze(torch.from_numpy(badfields[selidx, ...]), 1))

    samplelist = [realsamps, badsamps]
    betas = hf_diffusion.cosine_beta_schedule(1000)
    model = hf_diffusion.Unet(64, channels=1, dim_mults=(1, 2, 4)) #this alone is an uninitialized network lol
    sdict = torch.load(os.path.join(resdir, run,'/model.pt', map_location='cpu'))
    model.load_state_dict(sdict['model_state_dict'])

    diff = hf_diffusion.Diffusion(betas=betas)
    loss_type = 'huber'
    timesteps_eval = np.array([0, 249, 499, 749, 999])
    numnoise = 4
    loss_list = get_loss_for_samples(model, diff, loss_type, timesteps_eval, samplelist, numnoise)

    '''

    # Part 2: Looking at the losses of the best case baseline model: (where the ps is nearly the same)
    # the loss for hte earlier timesteps (nearer the data dbn) is MUCH worse for both the real images and the samples
    # the samples also had a lower loss for the earlier timesteps
    # i.e. if your schedule prioritized minimizing the loss of earlier steps that might be better
    #the loss near the data distribution is much higher than the loss near the noise end
    run = 'Run_8-23_2-9/'
    badfields = np.load(os.path.join(resdir, run)+'samples_100.npy')
    fields64 = np.load('../../data_processed/LogMaps_Mcdm_IllustrisTNG_LH_z=0.00_Nx64_train.npy')

    numimgs = 10
    rng = np.random.default_rng(seed=23)
    selidx = rng.choice(100, numimgs, replace=False)

    # transform
    RMIN, RMAX = torch.tensor(np.min(fields64)), torch.tensor(np.max(fields64))
    tr, invtr = hf_diffusion.get_minmax_transform(RMIN, RMAX)

    realsamps = tr(torch.unsqueeze(torch.from_numpy(fields64[selidx, ...]), 1))
    badsamps = tr(torch.unsqueeze(torch.from_numpy(badfields[selidx, ...]), 1))

    samplelist = [realsamps, badsamps]

    sdict = torch.load(os.path.join(resdir, run) +'model.pt', map_location='cpu')
    model_kwargs = sdict['model_kwargs']
    model = hf_diffusion.Unet(**model_kwargs)  #64, channels=1, dim_mults=(1, 2, 4)
    model.load_state_dict(sdict['model_state_dict'])
    #betas = hf_diffusion.linear_beta_schedule(2000, 1e-4, 0.02)
    betas = sdict['betas']

    diff = hf_diffusion.Diffusion(betas=betas)
    loss_type = 'huber'
    timesteps_eval = np.array([0, 499, 999, 1499, 1999])
    numnoise = 4
    loss_list = get_loss_for_samples(model, diff, loss_type, timesteps_eval, samplelist, numnoise)

    print(3)



