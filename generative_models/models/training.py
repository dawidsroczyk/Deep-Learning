# https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch 
import time
import torchvision.utils as vutils
import os
import numpy as np

def training(num_epochs, nz, dataloader, netD, netG, device, criterion, optimizerD, optimizerG, dir=None, real_label=1, fake_label=0):
    # Training Loop
    since = time.time()

    # Lists to keep track of progress
    # img_list = []
    G_losses = []
    D_losses = []
    D_real_mean_out = []
    D_fake_mean_out = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Save mean scores for plotting later
            D_real_mean_out.append(D_x)
            D_fake_mean_out.append(D_G_z1)
            
            # # Check how the generator is doing by saving G's output on fixed_noise
            # if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            #     with torch.no_grad():
            #         fake = netG(fixed_noise).detach().cpu()
            #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if dir != None:
        if not os.path.exists(os.path.join(dir, 'nets')):
            os.mkdir(os.path.join(dir, 'nets'))
        torch.save(netD.state_dict(), os.path.join(os.path.join(dir, 'nets'), 'netD'))
        torch.save(netG.state_dict(), os.path.join(os.path.join(dir, 'nets'), 'netG'))

        if not os.path.exists(os.path.join(dir, 'loss')):
            os.mkdir(os.path.join(dir, 'loss'))
        np.savetxt(os.path.join(os.path.join(dir, 'loss'), 'netD.txt'), np.array(D_losses))
        np.savetxt(os.path.join(os.path.join(dir, 'loss'), 'netG.txt'), np.array(G_losses))

        if not os.path.exists(os.path.join(dir, 'mean_out')):
            os.mkdir(os.path.join(dir, 'mean_out'))
        np.savetxt(os.path.join(os.path.join(dir, 'mean_out'), 'netD_real.txt'), np.array(D_real_mean_out))
        np.savetxt(os.path.join(os.path.join(dir, 'mean_out'), 'netD_fake.txt'), np.array(D_fake_mean_out))