# X-Bridge

This project is based on 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Addition of a novel X-Bridge method. X-Bridge was developed as a cross-modal bridge method for _img2sketch_ and _sketch2img_
translation in heterogeneous face recognition tasks.

X-Bridge pipeline. E = encoder, G1;G2 = generators, D1;D2 = discriminators, 
z = latent space. Dotted line indicates L1 loss. xr is real input from the first domain, xf^ is
reconstructed fake image from the first domain, xf^ is translated fake image from the second
domain, xr^ is corresponding real image from the second domain. The translation path is on
the left, whereas, the reconstruction path on the right.

![XBridge](imgs/XBridge_structure.png)
