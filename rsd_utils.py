import torch
import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

class RSDSampler:
    def __init__(self, scheduler_type="ddim"):
        self.scheduler_type = scheduler_type
        if scheduler_type=="ddim":
            self.scheduler = DDIMScheduler()
        elif scheduler_type=="ddpm":
            self.scheduler = DDPMScheduler()
        else:
            raise NotImplementedError(f"{scheduler} scheduler not implemented")
        self.scheduler.num_inference_steps = 1000
        self.jump_len = 4 # 10
        self.jump_n_sample = 4 # 10
        self.scheduler.betas = self.scheduler.betas.to("cuda")
        self.scheduler.alphas = self.scheduler.alphas.to("cuda")
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to("cuda")
        self.betas = self.scheduler.betas

    def _get_rsd_schedule(self, t):
        """
        Time schedule for Resampled Score Distillation (RSD)
        Referenced from RePaint (https://arxiv.org/pdf/2201.09865.pdf)
        """
        cur_t = t
        ts = [cur_t]
        for i in range(self.jump_n_sample):
            # Reverse Diffusion
            ts += [cur_t - (i + 1) for i in range(self.jump_len)]
            cur_t = ts[-1]
            # Forward Diffusion
            ts += [cur_t + (i + 1) for i in range(self.jump_len)]
            cur_t = ts[-1]
        return ts
    
    def predict_noise(self, t, unet, x, text_embeddings, cross_attention_kwargs, gt, mask, guidance_scale):
        times = self._get_rsd_schedule(t)
        
        for t_cur, t_future in zip(times[:-1], times[1:]):
            ######## Reverse Diffusion ########
            if t_future < t_cur:
                # Variant Eq.8(a) in https://arxiv.org/abs/2201.09865 (by paper t should be t_cur)
                x = mask * x + (1 - mask) * gt
                # Predict noise
                if guidance_scale != 1.0:
                    x_input = torch.cat([x] * 2)
                    noise_pred = unet(x_input, t_cur, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = unet(x, t_cur, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
                
                # p(x_t-1|x_t)
                x = self.scheduler.step(noise_pred, t_cur, x).prev_sample

            ######## Forward Diffusion ########
            else:
                # q(x_t|x_t-1)
                # https://arxiv.org/pdf/2006.11239.pdf Eq.2, eta=0 for ddim
                beta = self.betas[t_future]
                if self.scheduler_type == "ddpm":
                    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * torch.randn_like(x)
                else:
                    x = torch.sqrt(1 - beta)  * x
        
        # Final noise prediction from resampled x
        if guidance_scale != 1.0:
            x_input = torch.cat([x] * 2)
            noise_pred = unet(x_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            return noise_pred
        else:
            noise_pred = unet(x, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
            return noise_pred

rsd_sampler = RSDSampler()


def predict_noise0_diffuser_repaint(unet, noised_latents, noised_gt, mask, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False):
    batch_size = noised_latents.shape[0]

    
    if lora_v:
        # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=latents.device, dtype=latents.dtype
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    # Convert inputs to half precision
    if half_inference:
        latents = latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = rsd_sampler.predict_noise(t, unet, noised_latents, text_embeddings[batch_size:], cross_attention_kwargs, noised_gt, mask, guidance_scale)
    else:
        # predict the noise residual
        noise_pred = rsd_sampler.predict_noise(t, unet, noised_latents, text_embeddings, cross_attention_kwargs, noised_gt, mask, guidance_scale)
        # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        # perform guidance
    noise_pred = noise_pred.float()
    return noise_pred

def sds_vsd_grad_diffuser_repaint(unet, noise, text_embeddings, t, unet_phi=None, guidance_scale=7.5, \
                        grad_scale=1, cfg_phi=1., generation_mode='sds', phi_model='lora', \
                            cross_attention_kwargs={}, multisteps=1, scheduler=None, lora_v=False, \
                                half_inference = False, latents=None, gt=None, mask=None):
    # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114
    unet_cross_attention_kwargs = {'scale': 0} if (generation_mode == 'vsd' and phi_model == 'lora' and not lora_v) else {}
    noised_gt = scheduler.add_noise(gt, noise, t)
    noised_latents = scheduler.add_noise(latents, noise, t)
    with torch.no_grad():
        # predict the noise residual with unet
        # set cross_attention_kwargs={'scale': 0} to use the pre-trained model
        noise_pred = predict_noise0_diffuser_repaint(unet, noised_latents, noised_gt, mask, text_embeddings, t, guidance_scale=guidance_scale, cross_attention_kwargs=unet_cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)

    if generation_mode == 'sds':
        # SDS
        grad = grad_scale * (noise_pred - noise)
        # grad = grad_scale * (noise_pred)  # SJC
        noise_pred_phi = noise
    elif generation_mode == 'vsd':
        with torch.no_grad():
            noise_pred_phi = predict_noise0_diffuser(unet_phi, noised_latents, text_embeddings, t, guidance_scale=cfg_phi, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, lora_v=lora_v, half_inference=half_inference)
        # VSD
        grad = grad_scale * (noise_pred - noise_pred_phi.detach())

    grad = torch.nan_to_num(grad)

    ## return grad
    return grad, noise_pred.detach().clone(), noise_pred_phi.detach().clone()

def phi_vsd_grad_diffuser(unet_phi, latents, noise, text_embeddings, t, cfg_phi=1., grad_scale=1, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False):
    loss_fn = torch.nn.MSELoss()
    # ref to https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py#L114
    # predict the noise residual with unet
    noise_pred= predict_noise0_diffuser(unet_phi, latents, text_embeddings, t, guidance_scale=cfg_phi, cross_attention_kwargs=cross_attention_kwargs, scheduler=scheduler, half_inference=half_inference)
    loss = loss_fn(noise_pred, noise)
    loss *= grad_scale
    return loss

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, guidance_scale=7.5, cross_attention_kwargs={}, scheduler=None, lora_v=False, half_inference=False):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if lora_v:
        # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=noisy_latents.device, dtype=noisy_latents.dtype
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    # Convert inputs to half precision
    if half_inference:
        noisy_latents = noisy_latents.clone().half()
        text_embeddings = text_embeddings.clone().half()
        latent_model_input = latent_model_input.clone().half()
    if guidance_scale == 1.:
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
    else:
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v:
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    noise_pred = noise_pred.float()
    return noise_pred