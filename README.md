# Can denoising diffusion probabilistic models generate realistic astrophysical fields?

This repository contains code for the experiments in Can denoising diffusion probabilistic models generate realistic astrophysical fields?, accepted at the Neurips Machine Learning and Physical Sciences workshop.

We use code blocks and architecture from 
*  Hugging Face's [The Annotated Diffusion model](https://huggingface.co/blog/annotated-diffusion)
*  [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
*  [Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion) 

# Experiments
Experiments were kept track of using the [Weights and Biases](https://wandb.ai/) framework. <br/>
Cold Dark Matter Density Fields
1. For the 64x64 runs, run `python main.py config/params64_alt.yaml`
2. For the 128x128 run, run `python main.py config/params128_blseed.yaml`

Images from SFD <br/>
3. Run `python main.py config/params_dustmm.yaml`

An earlier version of this repository contained forked code from [Improved Denoising Diffusion Models](https://github.com/openai/improved-diffusion).
