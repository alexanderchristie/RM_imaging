#Notebooks for small tests
import os  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
plt.rcParams['axes.facecolor']='w'
plt.rcParams['savefig.facecolor']='w'
import matplotlib as mpl
mpl.rcParams['figure.facecolor'] = 'white'
#imports and plotting function
import argparse
import mat73
import logging
import numpy as np
#import torchvision
from datetime import datetime
import os
import scipy.io
from torch.utils.data import DataLoader
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import math
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import torch
import sys
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import torch
from sklearn.preprocessing import StandardScaler
import Models as M
import Helpers as H
import time
import copy
from torch.func import vmap
from functorch.experimental import replace_all_batch_norm_modules_
from sklearn.cluster import AgglomerativeClustering


encoder_out='sigmoid'
cwd=os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






if not os.path.isdir('Models'):
    os.mkdir('Models')

save_direc=os.path.join(cwd,'Models')

#os.path.isdir()
#data_path=os.path.join(cwd,'Data/PNAS-lowcoh_regime_all_seeds/PNAS-lowcoh_regime_seed0')

data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed0')
#data_path=os.path.join(cwd,'Data/PNAS-highcoh_regime_all_seeds/PNAS-highcoh_regime_seed0')
#data_path=os.path.join(cwd,'Data/PNAS-lowcoh_regime_guassian_05perc_offgrid_all_seeds/PNAS-lowcoh_regime_guassian_05perc_offgrid_seed0')
#data_path=os.path.join(cwd,'Data/PNAS-regime_all_seeds/PNAS-regime_seed0')
data_type=data_path.split('/')[-2]

medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
print(medium.shape)
inners=medium.transpose().conjugate()@medium
print(inners.shape)
coherence=0
for i in inners:
    for j in i:
        if abs(j)>coherence and j<.99:
            coherence=abs(j)
print('coherence of data: ', coherence)
print('test')



column_list=[]
index_list_list=[]
STACKED=True
unlabeled_data=5000
betas=(.9, .999)  #momenumt parameters
DICTIONARY_size_R=1024*2 #size of dictionary*2
layers=[4096, 2048]


for EXP_NUM in range(0,1):
    SCALING_param='None'
    WDECAY=1e-2
    ENC_GELMA_OPTIM='AdamW'
    DEC_OPTIM='AdamW'
    encoder_type='fc'
    decoder_init='None'
    SAV_GROUP='2Medium'


    data_type=data_path.split('/')[-2]
    if STACKED:
        data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed0')
        medium1= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
        data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed2')
        medium2= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
        medium=np.concatenate((medium1.T, medium2.T), axis=0)
        medium=medium.T




    else:
        medium= np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt'])
    inners=medium.transpose().conjugate()@medium
    coherence=0
    for i in inners:
        for j in i:
            if abs(j)>coherence and j<.99:
                coherence=abs(j)
    index_list=[]
    torch.cuda.empty_cache()
    torch.manual_seed(EXP_NUM)
    #Raw data unlabeled 
    starttime=time.time()
    target='raw'
    batchsize=128
    L1_weight_og=1e-1
    L1_weight=L1_weight_og
    L1_weight_final=L1_weight_og
    THRESH_HOLD_VAL=.001
    #[500,500,500,300,300,400,800]#[3000,1000,500,300,300,300,300,300,300,300,300,300,300,300]#[3000,1500,1000,500,500,500,300,300,300,400,400,800]#[1000,1000,1000,500,500,500,300,300,300,400,400,400,800]
    GELMA_layers=layers#[500,500,500,300,300,400,800]
    CE=False
    KM_in=False
    INV_weight=0
    RESHUFFLE=False
    LR=1e-3
    LR_final=LR/10
    GELMA_OG=0
    GELMA=GELMA_OG
    GELMA_Final=GELMA_OG
    TIME_reverse=False
    E_list=0
    RESETS=0
    Epochs=100
    index_liskillt=[]
    GELMA_inc=0
    G_0=False
    Scheduling=True  #NEED TO CHANGE TO FALSE TO NOT SCHEDULE
    END_schedule=1000
    dateTimeObj = datetime.now()
    if SCALING_param=='All':
        data_scaling=True
        GELMA_scaling=True
        Dec_scaling=True
    elif SCALING_param=='None':
        data_scaling=False
        GELMA_scaling=False
        Dec_scaling=False
    elif SCALING_param=='Weird':
        data_scaling=False
        GELMA_scaling=True
        Dec_scaling=True
    elif SCALING_param=='GELMA_only':
        GELMA_scaling=True
        data_scaling=False
        Dec_scaling=False
    elif SCALING_param=='GELMA_Error':
        GELMA_scaling='Error'
        data_scaling=False
        Dec_scaling=False
    elif SCALING_param=='Weird_rho':
        GELMA_scaling=True
        data_scaling=False
        Dec_scaling=True

    timestampStr = dateTimeObj.strftime("%H-%M-%S") 
    print('Timestamp: ', timestampStr)
    figsize,font_size=5, 25
    def soft_threh(x, th):
        x=x.squeeze()
        real, imag=torch.split(x, int(x.shape[-1]/2), dim=-1)
        modulus=torch.sqrt(real**2+imag**2)
        modulus=relu(modulus-th)
        theta=torch.atan2(imag, real).squeeze()
        real_out=modulus*torch.cos(theta)
        imag_out=modulus*torch.sin(theta) 
        return torch.cat((real_out,imag_out),-1) #relu(torch.abs(x)-th)
    #data_path=os.path.join(cwd,'Data/PNAS-regime_all_seeds/PNAS-regime_seed0')
    #data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed0')
    #data_path=os.path.join(cwd,'Data/PNAS-highcoh_regime_all_seeds/PNAS-highcoh_regime_seed0')
    print(medium.shape)
    inners=medium.transpose().conjugate()@medium
    print(inners.shape)
    coherence=0
    for i in inners:
        for j in i:
            if abs(j)>coherence and j<.99:
                coherence=abs(j)
    print('coherence of data: ', coherence)


    Track_run=True
    #Enter your wanbd key below and uncomment wanbd code to track run on wandb
    if Track_run:
        key='89a70fbc572a495206df640bd6c9cbf2a4a0dcaa' #  Novikov Key='f3c30b57562c6f42d1c8a091e13a45adf55424d3'
        wandb.login(key=key) 
        if DICTIONARY_size_R=='default':
            dict_sting='default'
        else:
            dict_sting=str(DICTIONARY_size_R/2)
        wandb.init(project='final sets', name=f'Scheduled. M0 sand M2, dictionary elements={dict_sting}. Stacked={STACKED}, exp num={EXP_NUM}')
        #wandb.init(project='rtt unlabeled', name=f'F 2 real. L1={L1_weight_og}, LR={LR}, dictionary elements={dict_sting}.')
        
        wandb.config['L1 final weight']=L1_weight_final
        wandb.config['unlabeled_data']=unlabeled_data 
        wandb.config['L1_weight']=L1_weight
        wandb.config['layers']=layers
        wandb.config['GELMA']=GELMA
        wandb.config['G_0']=G_0
        wandb.config['GELMA_layers']=GELMA_layers   
        wandb.config['inv_weight']=INV_weight
        wandb.config['CE']=CE
        wandb.config['timestamp']=timestampStr
        wandb.config['dcitonary size']=DICTIONARY_size_R
        wandb.config['data_path']=data_path
        wandb.config['EXP_NUM']=EXP_NUM
        wandb.config['Epochs']=Epochs
        wandb.config['betas']=betas
        wandb.config['LR']=LR
        wandb.config['LR_final']=LR_final
        wandb.config['data_type']=data_type
        wandb.config['SCALING_param']=SCALING_param
        wandb.config['GELMA_scaling']=GELMA_scaling
        wandb.config['Dec_scaling']=Dec_scaling
        wandb.config['data_scaling']=data_scaling
        wandb.config['weight decay']=WDECAY
        wandb.config['encoder_type']=encoder_type
        wandb.config['encoder and GELMA opt']=ENC_GELMA_OPTIM
        wandb.config['decoder opt']=DEC_OPTIM
        wandb.config['decoder_init']=decoder_init
        wandb.config['STACKED']=STACKED
        wandb.config['save_group']=SAV_GROUP
        wandb.config['Scheduling']=Scheduling
        wandb.config['END_schedule']=END_schedule
        
        
    wand_dict={}
    sigmoid = nn.Sigmoid()
    softmax=nn.Softmax(dim=0)

    

    if 'perc' in data_path:
        training_data=H.data_rho_pregen(data_path+'/train',unlabeled_data/80000)
        val_data=H.data_rho_pregen(data_path+'/val', 3000/80000)
    elif STACKED:
        data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed0')
        training_data=H.data_rho_loaded(data_path+'/train',unlabeled_data/2/80000,pixels='Gaussian_abs', normalize=data_scaling)
        val_data=H.data_rho_loaded(data_path+'/val', 3000/80000, pixels='Gaussian_abs', normalize=data_scaling)
        data_path=os.path.join(cwd,'Data/FoldyLox_all_seeds/FoldyLox_seed2')
        training_data2=H.data_rho_loaded(data_path+'/train',unlabeled_data/2/80000,pixels='Gaussian_abs', normalize=data_scaling)
        val_data2=H.data_rho_loaded(data_path+'/val', 3000/80000, pixels='Gaussian_abs', normalize=data_scaling)
        training_data.b=torch.cat((training_data.b, training_data2.b), dim=0)
        training_data.rho=torch.cat((training_data.rho, training_data2.rho), dim=0)
        val_data.b=torch.cat((val_data.b, val_data2.b), dim=0)
        val_data.rho=torch.cat((val_data.rho, val_data2.rho), dim=0)
    else:
        training_data=H.data_rho_loaded(data_path+'/train',unlabeled_data/80000,pixels='Gaussian_abs', normalize=data_scaling)
        val_data=H.data_rho_loaded(data_path+'/val', 3000/80000, pixels='Gaussian_abs', normalize=data_scaling)

    training_data.b=training_data.b.to(device)
    training_data.rho=training_data.rho.to(device)
    b_val=val_data.b.to(device)
    rho_val=val_data.rho.to(device)
    b_val=b_val.squeeze()
    xp=int((rho_val.shape[-1]/2)**(1/2))
    yp=xp
    indim=int(training_data.b[0].shape[0]/2)
    outdim=(training_data.rho[0].shape[0]/4)
    enc_dim=training_data.b[0].shape[0]/2
    if KM_in:
        enc_dim=enc_dim+outdim*2
    decoder_hats=[]

    if DICTIONARY_size_R=='default':
        enc_out=outdim*2
        dec_in=int(training_data.rho[0].shape[0])
    else:
        enc_out=DICTIONARY_size_R/2
        dec_in=DICTIONARY_size_R
    if encoder_type=='fc':
        encoder=M.fc_net_extra(enc_dim, layers,enc_out, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None,dropout=.5)
    elif encoder_type=='channeled':
        encoder=M.channeled_lin_layers_avg(enc_dim, layers, 3,int(enc_out), dropout=.5)

    decoder=M.norm_linear_complex(dec_in, int(training_data.b[0].shape[0]), normalize=Dec_scaling)  #R^ {N_k} -> C^{N_rec*N_freq}
    if decoder_init!='None':
        decoder.load_state_dict(torch.load(decoder_init))


    if Track_run:
        wandb.config['encoder_params']=H.count_parameters(encoder)
    print(H.count_parameters(encoder))


    if G_0:
        G_0_w=(np.array(mat73.loadmat(data_path+'/G_0.mat')['A0']))
        
        #G_0_w=(np.array(mat73.loadmat(data_path+'/rtt.mat')['Artt']))
        
        G_0_w=torch.cat((torch.tensor(G_0_w.real), torch.tensor(G_0_w.imag)), dim=0)
        G_0_w=G_0_w.float()
        decoder.weight.data=nn.parameter.Parameter(G_0_w.clone().detach().requires_grad_(True))
        Complex_eye=torch.cat((torch.eye(int(outdim*2)), torch.zeros((int(outdim*2),int(outdim*2)))), dim=1)

        medium_hat=decoder(Complex_eye).squeeze()
        medium_hat=F.normalize(medium_hat, dim=-1)
        print(f'G_0 initialization successful: {torch.allclose(medium_hat, G_0_w.T)}')


    print(H.count_parameters(encoder))
    if Track_run:
        wandb.config['encoder params']=H.count_parameters(encoder)
    print('Data shapes:', training_data.b[0].shape[0]/2, training_data.rho[0].shape[0]/4)
    encoder.to(device)
    decoder.to(device)
    relu=nn.ReLU()
    leaky_relu=nn.LeakyReLU(THRESH_HOLD_VAL)
    if ENC_GELMA_OPTIM=='AdamW':
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR,betas=betas, weight_decay=WDECAY)
    if ENC_GELMA_OPTIM=='AdamW':
        optimizer_decod = torch.optim.AdamW(decoder.parameters(), lr=LR,betas=betas, weight_decay=WDECAY)
    elif DEC_OPTIM=='SGD':
        optimizer_decod = torch.optim.SGD(decoder.parameters(), lr=LR, weight_decay=0.0)

    tanh=nn.Tanh()
    softmax=nn.Softmax(dim=1)
    bce_loss_fn=nn.BCELoss() 
    Threshold=nn.Threshold(THRESH_HOLD_VAL, 0)  
    L2_loss_fn=nn.MSELoss()
    CE_loss_fn=nn.CrossEntropyLoss()
    # if GELMA>0:
    #    L2_loss_fn=lambda x,y: torch.sqrt(nn.MSELoss()(x,y))
    L1_loss_fn=nn.L1Loss()
    if DICTIONARY_size_R=='default':
        Complex_eye=torch.cat((torch.eye(int(outdim*2)), torch.zeros((int(outdim*2),int(outdim*2)))), dim=1).to(device)

    else:
        Complex_eye=torch.cat((torch.eye(int(DICTIONARY_size_R/2)), torch.zeros((int(DICTIONARY_size_R/2),int(DICTIONARY_size_R/2)))), dim=1).to(device)


    softmax=nn.Softmax(dim=1)
    if GELMA>0:
        if encoder_type=='fc':
            GELMA_net=M.fc_net_extra(training_data.b[0].shape[0]/2, GELMA_layers, training_data.b[0].shape[0]/2, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None, dropout=.5)
        
        
        elif encoder_type=='channeled':
            GELMA_net=M.channeled_lin_layers_avg(training_data.b[0].shape[0]/2, GELMA_layers, 3,int(training_data.b[0].shape[0]/2), dropout=.5)

        
        if ENC_GELMA_OPTIM=='AdamW':
            optimizer_GELMA = torch.optim.AdamW(GELMA_net.parameters(), lr=min(LR/GELMA,1e0), maximize=True, betas=betas,  weight_decay=GELMA*(WDECAY)/(min(LR/GELMA,1e0)))
            #optimizer_GELMA = torch.optim.AdamW(GELMA_net.parameters(), lr=LR, maximize=True, betas=betas)

        
#            optimizer_GELMA = torch.optim.AdamW(GELMA_net.parameters(), lr=LR, maximize=True,betas=betas)
        lr_scheduler_GELMA=torch.optim.lr_scheduler.ExponentialLR(optimizer_GELMA, gamma=(LR_final/LR)**(1/END_schedule), last_epoch=-1)
        GELMA_net.to(device)
        #GELMA_net=nn.DataParallel(GELMA_net)
        GELMA_net.train()


    lr_scheduler_enc=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_final/LR)**(1/END_schedule), last_epoch=-1)
    lr_scheduler_decod=torch.optim.lr_scheduler.ExponentialLR(optimizer_decod, gamma=(LR_final/LR)**(1/END_schedule), last_epoch=-1)
    trainloader=DataLoader(training_data,batch_size=batchsize,shuffle=True,num_workers=0)
    if STACKED:
        outdim=outdim*2

    for epoch in range(Epochs):
        if RESETS>0 and epoch%RESETS==0:
            encoder=M.fc_net_extra(enc_dim, layers,outdim, net_type='fc',linear_type='real', activation='leaky', bias=True, out_scaling=None,dropout=.5)
            encoder.to(device)
            optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR)
            optimizer_decod = torch.optim.AdamW(decoder.parameters(), lr=LR)
        train_lossavg=0
        train_lossavg=0
        L1_loss_avg=0
        coupled_grad=0
        project_grad=0
        INV_loss_avg=0
        CEloss_avg=0


        inner_loss_term_avg=0
        for batch, (b, rho, num_targets) in enumerate(trainloader):
            b=b.to(device)
            rho=rho.to(device)
            b=b.squeeze()
            optimizer.zero_grad()
            optimizer_decod.zero_grad()
            rho_hat=encoder(b)
            rho_hat=soft_threh(rho_hat, THRESH_HOLD_VAL)
            if SCALING_param=='Weird_rho':
                rho_hat=F.normalize(rho_hat, dim=-1)

            b_hat=decoder(rho_hat)
            if L1_weight>0:
                L1_loss=L1_weight*L1_loss_fn(rho_hat, rho_hat*0)
                L1_loss.backward(retain_graph=True)  
                L1_loss_avg+=L1_loss.item()/L1_weight/len(trainloader)
            else:
                L1_loss=-1
            L2_loss=L2_loss_fn(b_hat.squeeze(), b.squeeze())
            L2_loss.backward(retain_graph=True)
            train_lossavg+=L2_loss.item()/len(trainloader)

            #if GELMA==0:
            
            if GELMA==0:
                optimizer.step()
                optimizer_decod.step()
                inner_loss_term_avg=-1


            elif GELMA>0 and epoch>0:
                GELMA_out=GELMA_net(b).squeeze()
                GELMA_out=soft_threh(GELMA_out, THRESH_HOLD_VAL)
                b_hat=b_hat.squeeze()
                error=b-b_hat
                if GELMA_scaling=='Error':
                    error=F.normalize(error, dim=-1)
                    GELMA_out=F.normalize(GELMA_out, dim=-1)
                elif GELMA_scaling==True:
                    GELMA_out=F.normalize(GELMA_out, dim=-1)
                
                
                
                GELMA_inners=torch.inner((error).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1)
                GELMA_inners=sum(GELMA_inners)/(torch.numel(GELMA_inners))/b.shape[-1]
                
                GELMA_inners=GELMA*(GELMA_inners)
                GELMA_inners.backward(retain_graph=True)

                optimizer.step()
                optimizer_decod.step()
                optimizer_GELMA.step()

                inner_loss_term_avg+=(GELMA_inners).item()/len(trainloader)/GELMA

                optimizer_GELMA.zero_grad()
                optimizer.zero_grad()
                optimizer_decod.zero_grad()


        
        L1_weight=min(L1_weight*(L1_weight_final/L1_weight_og)**(1/(END_schedule)), L1_weight_final)
        if GELMA>0:
            GELMA=min(GELMA*((GELMA_Final/GELMA_OG)**(1/(END_schedule))), GELMA_Final)
        if Scheduling>0 and epoch<END_schedule:
            lr_scheduler_enc.step()
            lr_scheduler_decod.step()
            if GELMA>0:
                lr_scheduler_GELMA.step()
        val_lossavg=0
        
        if DICTIONARY_size_R=='default':
            rho_hat=torch.abs(H.cat2complex(rho_hat.squeeze()))

            wand_dict['Training image']=H.plot_2_unordered_imgs(epoch,rho, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=xp, ypix2=yp, font_size=25)

        else:
            rho_hat=torch.abs(H.cat2complex(rho_hat.squeeze()))
            wand_dict['Training image']=H.plot_2_unordered_imgs(epoch,rho, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=int((DICTIONARY_size_R/2)**(1/2)), ypix2=int((DICTIONARY_size_R/2)**(1/2)), font_size=25)

        encoder.eval()
        decoder.eval()



        if INV_weight>0 and INV_weight<1e-1:
            INV_weight=INV_weight*(1.03)


        optimizer.zero_grad()
        optimizer_decod.zero_grad()

        font_size = 50
        if KM_in:
            medium_hat=decoder(Complex_eye).squeeze()
            medium_hat=F.normalize(medium_hat, dim=-1)
            Mhat_real, Mhat_imag=torch.split(medium_hat, int(medium_hat.shape[1]/2), dim=-1)
            Mhat_imag=-Mhat_imag
            b_real, b_imag=torch.split(b_val.squeeze(), int(b_val.shape[-1]/2), dim=-1)
            km_b_real=Mhat_real.squeeze()@b_real.squeeze().T-Mhat_imag.squeeze()@b_imag.squeeze().T
            km_b_imag=Mhat_real.squeeze()@b_imag.squeeze().T+Mhat_imag.squeeze()@b_real.squeeze().T

            km_b=torch.cat((km_b_real.T, km_b_imag.T), dim=-1)
            
            In_put=torch.cat((b_val, km_b), dim=-1)

            rho_hat=encoder(In_put)
        else:
            rho_hat=encoder(b_val)
            rho_hat=soft_threh(rho_hat, THRESH_HOLD_VAL)

            #rhos_hat=relu(abs(rho_hat)-.005)*torch.sign(rho_hat)            
        if SCALING_param=='Weird_rho':
            rho_hat=F.normalize(rho_hat, dim=-1)


        b_hat=decoder(rho_hat) 

        L2_loss=L2_loss_fn(b_hat.squeeze(), b_val.squeeze())
        val_lossavg+=L2_loss.item()
        if L1_weight>0:
            L1_loss=L1_weight*L1_loss_fn(rho_hat, rho_hat*0)
            val_lossavg+=L1_loss.item()
        if GELMA>0:
            GELMA_out=GELMA_net(b).squeeze()
            GELMA_out=soft_threh(GELMA_out, THRESH_HOLD_VAL)
            b_hat=b_hat.squeeze()
            error=b_val-b_hat
            if GELMA_scaling=='Error':
                error=F.normalize(error, dim=-1)
                GELMA_out=F.normalize(GELMA_out, dim=-1)
            elif GELMA_scaling:
                GELMA_out=F.normalize(GELMA_out, dim=-1)
            
            
            GELMA_inners=torch.inner((error).squeeze(),GELMA_out.squeeze()).diagonal(dim1=-2, dim2=-1)
            GELMA_inners=sum(GELMA_inners)/(torch.numel(GELMA_inners))/b.shape[-1]
            
            GELMA_inners=GELMA*(GELMA_inners)
            val_lossavg+=(GELMA_inners).item()








        if DICTIONARY_size_R=='default':
            rho_hat=torch.abs(H.cat2complex(rho_hat.squeeze()))

            wand_dict['Validation image']=H.plot_2_unordered_imgs(epoch, rho_val, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=xp, ypix2=yp, font_size=25)
        else:
            rho_hat=torch.abs(H.cat2complex(rho_hat.squeeze()))
            wand_dict['Num_indices_used']=len(set(rho_hat.nonzero(as_tuple=True)[1].cpu().detach().numpy()))
            wand_dict['Validation image']=H.plot_2_unordered_imgs(epoch, rho_val, rho_hat,ind=5, figsize=5, scaling=None, xpix=xp, ypix=yp, xpix2=int((DICTIONARY_size_R/2)**(1/2)), ypix2=int((DICTIONARY_size_R/2)**(1/2)), font_size=25)
        if epoch%1==0:
            
            sum_max_inner_original=0
            medium_hat=decoder(Complex_eye).squeeze()
            medium_hat=F.normalize(medium_hat, dim=-1)
            medium_hat=medium_hat.cpu().detach().numpy()
            medium_hat=H.cat2complex(medium_hat)                    
            torch_inners=np.abs(np.inner(medium_hat,medium.T.conj()) ) #entry i j is ghat_i dot bar(g_j) 

            index_list=[]
            for i in range(int(outdim*2)):
                if max(torch_inners[:, i])>.99 and i not in index_list:
                    index_list.append(i)
                sum_max_inner_original=sum_max_inner_original+max(torch_inners[:, i])
            max_avg_inners_original=sum_max_inner_original/(int(outdim*2))

            index_list_outs=[]
            for i in range(int(outdim*2)):
                if max(torch_inners[:, i])<.95 and i not in index_list_outs:
                    index_list_outs.append(i)
                

            maxG_inner_list=[max(torch_inners[:, i]) for i in range(int(outdim*2))]
            

            wand_dict['histogram']=H.hist_2_wandb(maxG_inner_list, figsize=5, font_size=25, epoch=epoch)
            wand_dict['Loss sum']=train_lossavg+L1_loss_avg*L1_weight+inner_loss_term_avg*GELMA
            wand_dict['train loss']=train_lossavg
            wand_dict['val loss']=val_lossavg
            if L1_loss>0:
                wand_dict['L1 loss']=L1_loss_avg
            if GELMA>0:
                wand_dict['GELMA loss']=inner_loss_term_avg
            if CE>0:
                wand_dict['CE loss']=CEloss_avg
            if INV_weight>0:
                wand_dict['INV loss']=INV_loss_avg
            if epoch%1000==0:
                torch.save(encoder.state_dict(), f'{save_direc}/encoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')    
                torch.save(decoder.state_dict(), f'{save_direc}/decoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
                if GELMA>0:    
                    torch.save(GELMA_net.state_dict(), f'{save_direc}/GELMA_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
            if epoch==4999:
                torch.save(encoder.state_dict(), f'{save_direc}/encoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')    
                torch.save(decoder.state_dict(), f'{save_direc}/decoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')

                if GELMA>0:    
                    torch.save(GELMA_net.state_dict(), f'{save_direc}/GELMA_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
            

        
            wand_dict['max avg inners']=max_avg_inners_original
            wand_dict['num indices']=len(index_list)
            wand_dict['num indices <.95']=len(index_list_outs)
        if Track_run:
            wandb.log(wand_dict)
        

    print(time.time()-starttime)
    if Track_run:
        wandb.finish()
    index_list_list.append(index_list)
    medium_hat=decoder(Complex_eye).squeeze()
    medium_hat=F.normalize(medium_hat, dim=-1)
    medium_hat=medium_hat.cpu().detach().numpy()
    medium_hat=H.cat2complex(medium_hat)                    
    column_list.append(medium_hat)
    torch.save(encoder.state_dict(), f'{save_direc}/encoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')    
    torch.save(decoder.state_dict(), f'{save_direc}/decoder_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')
    if GELMA>0:    
        torch.save(GELMA_net.state_dict(), f'{save_direc}/GELMA_{unlabeled_data}_{EXP_NUM}_{timestampStr}.pt')








