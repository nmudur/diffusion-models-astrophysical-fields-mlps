
#https://huggingface.co/blog/annotated-diffusion

import os
import sys
import wandb
import yaml
import torch
import datetime
import numpy as np
import astropy

import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, CenterCrop, Resize
from astropy.io import fits
import matplotlib.pyplot as plt
from functools import partial
import pickle

import hf_diffusion
from hf_diffusion import *
import utils


with open(sys.argv[1], 'r') as stream:
    config_dict = yaml.safe_load(stream)

if 'seed' in config_dict.keys():
    SEED = int(config_dict['seed'])
else:
    SEED = 23 #5 for all older runs

torch.manual_seed(SEED)
np.random.seed(SEED)

DEBUG= False

dt = datetime.datetime.now()
name = f'Run_{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'

print(name)

timesteps = int(config_dict['diffusion']['timesteps'])
epochs = int(config_dict['train']['epochs'])
beta_schedule_key = config_dict['diffusion']['beta_schedule']
DATAPATH = config_dict['data']['path']


BATCH_SIZE = int(config_dict['train']['batch_size'])
LR = float(config_dict['train']['learning_rate'])

if torch.cuda.is_available(): 
    device = 'cuda'
else: 
    device='cpu'
print(device)

#moving to diffusion
beta_func = getattr(hf_diffusion, beta_schedule_key)
beta_args = config_dict['diffusion']['schedule_args']
beta_schedule = partial(beta_func, **beta_args)
betas = beta_schedule(timesteps=timesteps)
diffusion = Diffusion(betas)




def train(model, dataloader, optimizer, epochs, loss_type="huber", sampler=None, conditional=False, resdir=None,
          misc_save_params=None, inverse_transforms=None, start_itn=0, start_epoch=0):

    '''
    #alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) #needed where?
    #sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    #posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    '''
    itn = start_itn
    epoch = start_epoch
    loss_spike_flg = 0
    while epoch<epochs:  # Epochs: number of full passes over the dataset
        print('Epoch: ', epoch)
        for step, batch in enumerate(dataloader):  # Step: each pass over a batch
            optimizer.zero_grad() #prevents gradient accumulation
            if conditional:
                batch, labels = batch
                labels = labels.to(device)

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Algorithm 1 line 3: sample t uniformly for every example in the batch
            t = sampler.get_timesteps(batch_size, itn) #[0, T-1]
            loss = diffusion.p_losses(model, batch, t, loss_type=loss_type, labels=labels if conditional else None)
            if sampler.type=='loss_aware':
                with torch.no_grad():
                    loss_timewise = diffusion.timewise_loss(model, batch, t, loss_type=loss_type, labels=labels if conditional else None)
                    sampler.update_history(t, loss_timewise)
            if step % 100 == 0:
                print("Loss:", loss.item())
            if not DEBUG:
                wandb.log({"loss": loss.item(), "iter": itn, "epoch": epoch})
            if loss.item()>0.1 and itn>300 and (loss_spike_flg<2):
                badbdict = {'batch': batch.detach().cpu().numpy(), 'itn': itn, 't': t.detach().cpu().numpy(), 'loss': loss.item()}
                pickle.dump(badbdict, open(resdir+f'largeloss_{itn}.pkl', 'wb'))
                loss_spike_flg+=1

            loss.backward()

            optimizer.step()
            itn+=1
            if itn%2000 == 0:
                misc_save_params.update({'epoch': epoch, 'itn': itn, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()})
                torch.save(misc_save_params, resdir+f'checkpoint_{itn}.pt')
                #save samples
                if itn%2000 == 0:
                    NSAMP = 10
                    samples = diffusion.sample(model, image_size=misc_save_params["model_kwargs"]["dim"], batch_size=NSAMP,
                                               channels=misc_save_params["model_kwargs"]["channels"])
                    sampimg = np.stack([samples[-1][i].reshape(image_size, image_size) for i in range(NSAMP)])
                    invtsamples = inverse_transforms(torch.tensor(sampimg)).numpy()
                    np.save(resdir+f'samples_{itn}.npy', invtsamples)
        epoch+=1
    return itn, epoch



def get_default_model_kwargs(image_size, channels, config_dict):
    dim_mults = tuple([int(elem) for elem in config_dict['architecture']['dim_mults']])
    dim = image_size if 'unet_dim' not in config_dict['architecture'] else int(config_dict['architecture']['unet_dim'])
    model_kwargs = {"dim": image_size, "channels": channels, "dim_mults": dim_mults, "use_convnext": False} #this is using dim=Nx by default
    model_kwargs.update({'init_conv': True if 'init_conv' not in config_dict['architecture'].keys() else bool(config_dict['architecture']['init_conv'])})
    if bool(config_dict['data']['conditional']):
        model_kwargs.update({'conditional_dim': int(config_dict['data']['conditional_dim'])})
    
    if 'time_embed_dim' in config_dict['architecture'].keys():
        model_kwargs.update({'time_embed_dim': int(config_dict['architecture']['time_embed_dim'])})
    return model_kwargs

def get_defaults_config_dict(config_dict):
    if 'conditional' not in config_dict['data']:
        config_dict['data'].update({'conditional': False})
    if 'loss_type' not in config_dict['train'].keys():
        config_dict['train'].update({'loss_type': 'huber'})
    if 'sampler_args' not in config_dict['diffusion'].keys():
        config_dict['diffusion'].update({'sampler_args': {'sampler_type': 'uniform'}})
    if 'transforms' not in config_dict['data'].keys():
        config_dict['data'].update({'transforms': 'minmax'}) #by default: scales the minmax of the training data to [-1, 1]
    return config_dict


if __name__ == '__main__':
    config_dict = get_defaults_config_dict(config_dict)
    CONDITIONAL = bool(config_dict['data']['conditional'])

    if not DEBUG:
        if 'resume_id' in config_dict.keys():
            wandb.init(project='diffmod_cosmo0', job_type='conditional' if CONDITIONAL else 'unconditional',
                       config=config_dict, name=name, id = config_dict['resume_id'], resume='must')
        else:
            wandb.init(project='diffmod_cosmo0', job_type='conditional' if CONDITIONAL else 'unconditional',
                       config=config_dict, name=name)

    ### get training data
    if 'fits' in DATAPATH:
        with fits.open(DATAPATH, memmap=True) as hdul:
            imgmemmap = hdul[0].data
    else:
        imgmemmap = np.load(DATAPATH, mmap_mode='r')

    image_size = imgmemmap[0].shape[-1]
    channels = 1
    if not DEBUG:
        wandb.config['data'].update({'image_size': image_size, 'channels': channels})
    NTRAIN = imgmemmap.shape[0]
    
    #retrieve data transformations
    if config_dict['data']['transforms']=='minmax':
        print('Minmax scaling to -1, 1')
        RANGE_MIN, RANGE_MAX = torch.tensor(imgmemmap.min()), torch.tensor(imgmemmap.max())
        transforms, inverse_transforms = get_minmax_transform(RANGE_MIN, RANGE_MAX)
    elif config_dict['data']['transforms']=='None':
        print('No prior data transformation applied')

        transforms, inverse_transforms = nn.Identity(), nn.Identity()
    elif config_dict['data']['transforms']=='center':
        print('Center so mean 0 but no multiplicative scaling')
        RANGE_MIN, RANGE_MAX = torch.tensor(imgmemmap.min()), torch.tensor(imgmemmap.max())
        transforms, inverse_transforms = get_center_transform(RANGE_MIN, RANGE_MAX)
    else:
        raise NotImplementedError()

    traindata = CustomTensorDataset(imgmemmap, transforms=transforms, labels_path=config_dict['data']['labels'] if CONDITIONAL else None)
    dataloader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)


    ### get model
    if config_dict['architecture']['model'] == 'baseline':
        model_kwargs = get_default_model_kwargs(image_size, channels, config_dict)
        model = Unet(**model_kwargs)
        model.to(device)
        if 'resume_id' in config_dict:
            sdpath = 'results/samples_exps/'+config_dict['resume_name']+'/'+config_dict['resume_ckp']
            sdict = torch.load(sdpath, map_location='cpu')
            print(sdict.keys())
            model.load_state_dict(sdict['model_state_dict'])
    else:
        raise NotImplementedError()

    if config_dict['train']['optimizer']=='Adam':
        if 'resume_id' in config_dict:
            optimizer = Adam(model.parameters())
            optimizer.load_state_dict(sdict['optimizer_state_dict'])
        else:
            optimizer = Adam(model.parameters(), lr=LR)
    elif  config_dict['train']['optimizer']=='SGD':
        if 'resume_id' in config_dict:
            optimizer = SGD(model.parameters())
            optimizer.load_state_dict(sdict['optimizer_state_dict'])
        else:
            optimizer = SGD(model.parameters(), lr=LR)

    else:
        raise NotImplementedError()

    #get sampler type
    #sampler_args =
    sampler = TimestepSampler(timesteps=timesteps, device='cuda', **config_dict['diffusion']['sampler_args'])

    #sample from trained model
    resdir = f'results/samples_exps/{name}/'
    os.mkdir(resdir)
    misc_save_params = {'model_type': config_dict['architecture']['model'],
                "model_kwargs": model_kwargs,
                "schedule": beta_schedule_key, "schedule_args": beta_args, "betas": betas}
    start_itn = 0 if 'resume_id' not in config_dict.keys() else sdict['itn']
    start_epoch = 0 if 'resume_id' not in config_dict.keys() else sdict['epoch']
    end_itn, end_epoch = train(model, dataloader, optimizer, epochs=epochs, loss_type=config_dict['train']['loss_type'], sampler=sampler, conditional=CONDITIONAL,
          resdir=resdir, misc_save_params=misc_save_params, inverse_transforms = inverse_transforms, start_itn=start_itn, start_epoch=start_epoch)
    
    misc_save_params.update({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'itn': end_itn, 'epoch': end_epoch})
    torch.save(misc_save_params, resdir+'model.pt')


    if CONDITIONAL:
        NSAMP = 20
        rng = np.random.default_rng()
        lidx = rng.choice(1000, 5, replace=False)
        samplabels = np.zeros((NSAMP, config_dict['data']['conditional_dim']), dtype=np.float32)
        for si in range(NSAMP):
            samplabels[si, :] = traindata.labels[lidx[si // 4], :]
        samplabels = torch.from_numpy(samplabels)
        samplabels = samplabels.to(device)
        samples = diffusion.sample(model, image_size=image_size, batch_size=NSAMP, channels=channels, labels=samplabels)
        np.savetxt(resdir+'samplabels.txt', samplabels.cpu().numpy())
    else:
        NSAMP=10
        samples = diffusion.sample(model, image_size=image_size, batch_size=NSAMP, channels=channels)

    for i in range(NSAMP):
        plt.figure()
        c = plt.imshow(samples[-1][i].reshape((image_size, image_size)), origin='lower')
        plt.colorbar(c)
        plt.savefig(resdir+f'{i}.png')
        plt.show()

    #get samples and real ps
    kvals = np.arange(0, image_size/2)

    #real
    rand_samp = np.random.choice(NTRAIN, NSAMP, replace=False)
    realimg = imgmemmap[rand_samp, :, :]
    realps = np.stack([utils.calc_1dps_img2d(kvals, realimg[i], smoothed=0.5) for i in range(NSAMP)])
    meanps, minps, maxps = np.mean(realps, axis=0), np.percentile(realps, 5, axis=0), np.percentile(realps, 95, axis=0)
    #minmax only actually percentiles when nsamp is 100ish
    #sampled
    sampimg = np.stack([samples[-1][i].reshape(image_size, image_size) for i in range(NSAMP)])
    invtsamples = inverse_transforms(torch.tensor(sampimg)).numpy()
    sampps = np.stack([utils.calc_1dps_img2d(kvals, invtsamples[i], smoothed=0.5) for i in range(NSAMP)])
    smeanps, sminps, smaxps = np.mean(sampps, axis=0), np.percentile(sampps, 5, axis=0), np.percentile(sampps, 95, axis=0)
    
    #plot
    plt.figure()
    plt.plot(kvals, meanps, label='Real')
    plt.fill_between(kvals, minps, maxps, alpha=0.2)
    plt.plot(kvals, smeanps, label='Samples')
    plt.fill_between(kvals, sminps, smaxps, alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig(resdir+'powspec.png')
    
    ps_rmse = np.sqrt(np.mean((meanps - smeanps)**2))

    if 'fits' in DATAPATH:
        hdul.close()

    if not DEBUG:
        wandb.log({"sample": wandb.Image(resdir+f'{i}.png'), "powspec": wandb.Image(resdir+'powspec.png'), "powspec_rmse": ps_rmse})

