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
import random
import torch.optim as optim
import mat73
import re
import copy
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.preprocessing import StandardScaler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




import pandas as pd

def read_and_concatenate(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Concatenate EXP_NUM and timestamp columns
    concatenated_list = (df['EXP_NUM'].astype(str) + '_' + df['timestamp'].astype(str)).tolist()
    
    return concatenated_list



#Data set class. Should contain your data, and when __getitem__ is called, it should 
#return your data formatted for input into a model
class data_rho_loaded:
    def __init__(self,data_path ,prop,sparsity=4,seed=0, pixels='One-hot',normalize=False):
        if  "train" in data_path:
            self.rho, self.b=Generate_data_pnas(data_path[:-5],int(80000*prop), S=sparsity,seed=seed,pixels=pixels )
            self.data_path=data_path[:-5]

        elif  'val' in data_path:
            self.rho, self.b=Generate_data_pnas(data_path[:-3],3000, S=sparsity,seed=100,pixels=pixels)
            self.data_path=data_path[:-3]
        self.rho=torch.cat((torch.tensor(self.rho.real),torch.tensor(self.rho.imag)),dim=-1).float()
        self.b=torch.cat((torch.tensor(self.b.real),torch.tensor(self.b.imag)),dim=-1).float()

        if prop<=1.0:
            self.rho=self.rho.to(device)
            self.b=self.b.to(device)
    
        if normalize:
            self.b=F.normalize(self.b, dim=-1)

    def Check_data(self,medium):
        rho=self.rho[:1000]
        b=self.b[:1000]
        rho=rho.cpu().detach().numpy()
        b=b.cpu().detach().numpy()
        rho=cat2complex(rho)
        b=cat2complex(b)
        if not np.allclose(medium@rho.T, b.T):
            print('Data Incorrect. Exiting code')
            exit()
        else:
            print('Data formatted correctly')


    def __len__(self):
        return(int(len(self.b)))
    def __getitem__(self, idx):
        return self.b[idx,...], self.rho[idx,...], torch.sum(self.rho[idx,...])


class data_rho_pregen(data_rho_loaded):
    def __init__(self,data_path ,prop,sparsity=4,seed=0, pixels='One-hot',normalize=False):
        if 'train' in data_path:
            self.data_path=data_path[:-5]
        elif 'val' in data_path:
            self.data_path=data_path[:-3]
            
        self.b=np.array(mat73.loadmat(self.data_path+'b.mat')['b'])
        self.rho=np.array(mat73.loadmat(self.data_path+'rho.mat')['rho'])
        if 'train' in data_path:
            self.b=self.b[:int(80000*prop)]
            self.rho=self.rho[:int(80000*prop)]
        elif 'val' in data_path:
            self.b=self.b[-500:]
            self.rho=self.rho[-500:]
        self.rho=torch.cat((torch.tensor(self.rho.real),torch.tensor(self.rho.imag)),dim=-1).float()
        self.b=torch.cat((torch.tensor(self.b.real),torch.tensor(self.b.imag)),dim=-1).float()

        if prop<=1.0:
            self.rho=self.rho.to(device)
            self.b=self.b.to(device)
    
        if normalize:
            self.b=F.normalize(self.b, dim=-1)







class data_rho_CC(data_rho_loaded):
    def __init__(self,data_path ,prop,sparsity=4, seed=0, medium='Random'):
        if 'train' in data_path:
            self.data_path=data_path[:-5]
        elif 'val' in data_path:
            self.data_path=data_path[:-3]



        if sparsity==1:
            self.rho, self.b=Generate_data_pnas_EYE(self.data_path, medium=medium)
        
        elif 'PNAS' in data_path and "train" in data_path:
            self.rho, self.b=Generate_data_pnas(data_path[:-5],int(80000*prop), S=sparsity,seed=seed)

        elif 'PNAS' in data_path and 'val' in data_path:
            self.rho, self.b=Generate_data_pnas(data_path[:-3],3000, S=sparsity,seed=100)


        
        self.Mask=np.array(mat73.loadmat(self.data_path+'/M.mat')['M'])
        self.rho=torch.cat((torch.tensor(self.rho.real),torch.tensor(self.rho.imag)),dim=-1).float()


        
    def __len__(self):
        return(int(len(self.b)))

    def __getitem__(self, idx):

        outer=np.outer(self.b[idx,...],self.b[idx,...].conj())
        outer=outer[abs(self.Mask)>0]
        outer=outer.ravel()
        return torch.cat((torch.tensor(outer.real),torch.tensor(outer.imag)),dim=-1).float(),self.rho[idx,...].float(), float(torch.sum(self.rho[idx,...].real))
    def get_data(self):
        outer_list=[]
        for idx in range(len(self.b)):
            outer=np.outer(self.b[idx,...],self.b[idx,...].conj())
            outer=outer[abs(self.Mask)>0]
            outer=outer.ravel()
            
            outer=torch.cat((torch.tensor(outer.real),torch.tensor(outer.imag)),dim=-1).float()
            outer_list.append(outer.squeeze())
        outer_list=torch.stack(outer_list)  
            
        return outer_list, self.rho
    



class data_rho_CC_XY_targs(data_rho_CC):
    def __init__(self,data_path ,prop,sparsity=4, seed=0, medium='Random', grid='grid.mat'):
        super().__init__(data_path, prop, sparsity, seed, medium)
        self.rho=np.array(mat73.loadmat(self.data_path+grid)['full_grid'])
        self.rho=torch.tensor(self.rho).float()
        self.intercept_x=self.rho[:,0].min()
        self.slope_x=(self.rho[:,0].max()-(self.rho[:,0].min()))
        self.intercept_y=(self.rho[:,1].min())
        self.slope_y=((self.rho[:,1].max())-(self.rho[:,1].min()))
        self.intercept_x=copy.deepcopy(self.intercept_x)
        self.slope_x=copy.deepcopy(self.slope_x)
        self.intercept_y=copy.deepcopy(self.intercept_y)
        self.slope_y=copy.deepcopy(self.slope_y)
        self.rho[:,0]=(self.rho[:,0]-min(self.rho[:,0]))/(max(self.rho[:,0])-min(self.rho[:,0]))
        self.rho[:,1]=(self.rho[:,1]-min(self.rho[:,1]))/(max(self.rho[:,1])-min(self.rho[:,1]))
    def __getitem__(self, idx):

        outer=np.outer(self.b[idx,...],self.b[idx,...].conj())
        outer=outer[abs(self.Mask)>0]
        outer=outer.ravel()
        return torch.cat((torch.tensor(outer.real),torch.tensor(outer.imag)),dim=-1).float(), self.rho[idx,...].float(), float(torch.sum(self.rho[idx,...].real))




class data_rho_CC_IID(data_rho_loaded):
    def __init__(self,data_path ,prop):
        self.Mask=np.array(mat73.loadmat(data_path+'/M.mat')['M'])

        self.b=np.load(data_path+'/b.npy') 
        self.rho=np.load(data_path+'/rho.npy')
        
        self.b=self.b[:int(80000*(prop))]
        self.rho=self.rho[:int(80000*(prop))]
        self.rho=torch.cat((torch.tensor(self.rho.real),torch.tensor(self.rho.imag)),dim=-1).float()
        

        
    def __len__(self):
        return(int(len(self.b)))

    def __getitem__(self, idx):
        
        outer=np.outer(self.b[idx,...],self.b[idx,...].conj())
        outer=outer[abs(self.Mask)>0]
        outer=outer.ravel()
        return torch.cat((torch.tensor(outer.real),torch.tensor(outer.imag)),dim=-1).float(),self.rho[idx,...].float(), float(torch.sum(self.rho[idx,...].real))
        


#Computes number correct if target is 1-hot
def accuracy(output, target):
    num_correct=sum(torch.argmax(output, dim=1)==torch.argmax(target, dim=1))
    return num_correct/len(target)  


class data_rho_presaved:
    def __init__(self,data_path ,prop):

        self.b=np.load(data_path+'/b.npy')
        self.rho=np.load(data_path+'/rho.npy')
        self.b=self.b[:int(len(self.b)*prop)]
        self.rho=self.rho[:int(len(self.rho)*prop)]
        self.rho=torch.cat((torch.tensor(self.rho.real),torch.tensor(self.rho.imag)),dim=-1).float()
        self.b=torch.cat((torch.tensor(self.b.real),torch.tensor(self.b.imag)),dim=-1).float()

    def Check_data(self,medium):
        rho=self.rho[:1000]
        b=self.b[:1000]
        rho=rho.cpu().detach().numpy()
        b=b.cpu().detach().numpy()
        rho=cat2complex(rho)
        b=cat2complex(b)
        if not np.allclose(medium@rho.T, b.T):
            print('Data Incorrect. Exiting code')
            exit()
        else:
            print('Data formatted correctly')


    def __len__(self):
        return(int(len(self.b)))
    def __getitem__(self, idx):
        return self.b[idx,...], self.rho[idx,...], float(torch.sum(self.rho[idx,...].real))


#Computes number correct if target is 1-hot
def accuracy(output, target):
    num_correct=sum(torch.argmax(output, dim=1)==torch.argmax(target, dim=1))
    return num_correct/len(target)  




#sends image to Wandb platform
def output2wandb(rho, rho_hat, i, epoch, xpix=31, ypix=21):
    plt.close()
    fig = plt.Figure(figsize=(40,40))
    plt.subplot(1,2,1)
    img=torch.abs(cat2complex(rho[i,:]).view(xpix,ypix).detach().cpu())
    plt.pcolor(img/torch.max(img),cmap='jet')
    cbar_used=plt.colorbar()
    plt.title('Target',fontsize=20)
    cbar_used.ax.tick_params(labelsize=20)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.subplot(1,2,2)
    
    
    
    img=torch.abs(cat2complex(rho_hat[i,:]).view(xpix,ypix).detach().cpu())
    plt.pcolor(img/torch.max(img),cmap='jet')
    plt.title('Output',fontsize=20)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.mappable.set_clim(*cbar_used.mappable.get_clim())

    fig = plt.gcf()
    fig.set_figheight(15)
    fig.set_figwidth(35)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
     
    del fig, buf
    return [wandb.Image(img, caption="Epoch "+str(epoch))]

    

# Counts number of params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#concatenates [Re(v), Im(v)]--> Re(v)+i Im(v)
def cat2complex(v):
    n=v.shape[-1]
    n=n//2
    return v[...,:n]+1j*v[...,n:]

#view for complex vectors
def complexview(x, rows):
    real, imag=torch.split(x, int(x.shape[-1]/2),dim=-1)
    real=real.view(real.shape[0], rows, int(real.shape[-1]/rows))
    imag=imag.view(imag.shape[0],  rows,int(imag.shape[-1]/rows))
    return torch.cat((real, imag), -2)

# complex mat mul for [Re(v), Im(v)]
def complex_mat_mul(M, x):
    real_mat, imag_mat=torch.split(M, int(M.shape[1]/2),dim=1)
    real_x, imag_x=torch.split(x, int(x.shape[-1]/2),dim=-1)


    real_out=(torch.matmul(real_mat,real_x.unsqueeze(-1))-torch.matmul(imag_mat, imag_x.unsqueeze(-1))).squeeze()
    imag_out=(torch.matmul(real_mat,imag_x.unsqueeze(-1))+torch.matmul(imag_mat, real_x.unsqueeze(-1))).squeeze()
    return torch.cat((real_out, imag_out), -1)




#gets intergers from string
def extract_integer(text):
    # Define a regular expression pattern to match integers
    
    return list(map(int, re.findall(r'\d+', text)))


#reformats data from matlab to numpy
def raw_data_reformat(datapath):
    try:
        os.mkdir(datapath+'/train')
        os.mkdir(datapath+'/val')
    except:
        pass
    
    torch.manual_seed(0)
    np.random.seed(0) 
    perm = np.random.permutation(5)
    loader=scipy.io.loadmat
    train_list=[]
    val_list=[]
    name_list=[]

    try:
        b1=np.array(mat73.loadmat(datapath+'/raw_train.mat')['b_d_train'])
    except:
        try:
            b1=np.array(mat73.loadmat(datapath+'/b_d_train.mat')['b_d_train'])
        except:
            b1=np.array(mat73.loadmat(datapath+'/b_train.mat')['b_d_train'])
            

    rho1=np.array(mat73.loadmat(datapath+'/rho_train.mat')['rho_d_train'])
    #rhoh1=np.array(mat73.loadmat(datapath+'/rhoh_d_train.mat')['rhoh_d_train'])
    cc1=np.array(mat73.loadmat(datapath+'/cc_train.mat')['cc_d_train'])
    
    train_list.append(b1)
    train_list.append(rho1)
    train_list.append(cc1)

    try:
        b_val=np.array(mat73.loadmat(datapath+'/b_val.mat')['b_d_val'])
    except:
        b_val=np.array(mat73.loadmat(datapath+'/raw_val.mat')['b_d_val'])

    rho_val=np.array(mat73.loadmat(datapath+'/rho_val.mat')['rho_d_val'])
    # rhoh_val=np.array(mat73.loadmat(datapath+'/rhoh_d_val.mat')['rhoh_d_val'])
    cc_val=np.array(mat73.loadmat(datapath+'/cc_val.mat')['cc_d_val'])

    val_list.append(b_val)
    val_list.append(rho_val)

    val_list.append(cc_val)
    name_list.append('b')
    name_list.append('rho')
    name_list.append('cc')

#    except: 
#       print('No raw, cc data found')
    for i in range(3):
        perm = np.random.permutation(len(b1))
        train_list=[i[perm] for i in train_list]


        perm = np.random.permutation(len(b_val))
        
        val_list=[i[perm] for i in val_list]



    for train, val, name in zip(train_list,val_list,name_list):
        if 'sameconfigurations' in datapath:
            np.save(f'{datapath}/val/{name}.npy',train[-3000:])
            print('saved tail of training data as val data')
        else:
            np.save(f'{datapath}/val/{name}.npy',val)
        np.save(f'{datapath}/train/{name}.npy',train)
    







def Get_ghat_elem(decoder):
    Complex_eye=torch.eye(651).unsqueeze(1).to(device)

    try:
        error, linear= decoder(Complex_eye)
        medium_hat=error+linear
    except:
        medium_hat=decoder(Complex_eye)

    medium_hat=medium_hat.squeeze()

    medium_hat=F.normalize(medium_hat, dim=-1)
    medium_hat=medium_hat.cpu().detach().numpy()
    medium_hat=cat2complex(medium_hat)
    
    return medium_hat





def Generate_data(locat, amount):
    medium= np.array(mat73.loadmat(locat+'/rtt.mat')['Artt'])
    S=locat.split('/')[4].split('_')[3][0]
    data_rho=np.zeros((amount,651))
    #data_b=np.zeros((amount,631))
    
    for i in range(amount):
        data_rho[i][:5]=1
        perm = np.random.permutation(651)
        data_rho[i]=data_rho[i][perm]
    print(medium.shape, data_rho.shape)
    data_b=medium@data_rho.T
    return data_rho, data_b.T

#plots images
def save_2_imgs(rho, rho_hat,ind=9,figsize=8,scaling='Linf',font_size=50,  xpix=31, ypix=21, file_name=None):
    plt.close()
    figsize=(figsize,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 1,figsize=figsize)

    tick_params = {'labelsize': font_size}
    output=rho.squeeze()
    ax=axes
    output=output[ind,:]
    if scaling=="Linf":
        output=output/torch.max(torch.abs(cat2complex(output)))
    elif scaling=="L1":
        output=output/torch.sum(torch.abs(cat2complex(output)))
    img=torch.abs(cat2complex(output.squeeze())).view(xpix, ypix)
    pcol1=ax.pcolor(img.detach().cpu(),cmap='jet')
    cbar_used=plt.colorbar(pcol1,ax=ax)

    cbar_used.ax.tick_params(labelsize=font_size)
    ax.tick_params(axis='both', **tick_params)
    cbar_used.remove()
        
    plt.savefig(f'/home/achristie/Codes_data/E_D_figs/{file_name}_true.pdf')

    plt.close()
    figsize=(figsize,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 1,figsize=figsize)

    output=rho_hat.squeeze() 
    output=output[ind,:]
    if scaling=="Linf":

        output=output/torch.max(torch.abs((output)))
    elif scaling=="L1":
        output=output/torch.sum(torch.abs((output)))
    img=torch.abs((output.squeeze())).view(xpix, ypix)
    
    ax=plt.gca()
    pcol2=ax.pcolor(img.detach().cpu(),cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
#plt.title('true', fontsize=font_size)
    cbar.mappable.set_clim(*cbar_used.mappable.get_clim())
    
    cbar.ax.tick_params(labelsize=font_size)
    cbar.remove()
    ax.tick_params(axis='both', **tick_params)
    plt.savefig(f'/home/achristie/Codes_data/E_D_figs/{file_name}_predicted.pdf')

    plt.close()

#different plotter
def plot_2_imgs(rho, rho_hat,ind=9,figsize=8,scaling='Linf',font_size=50, Single=False, xpix=31, ypix=21, file_name=None, Lclim=None, Hclim=None):
    plt.close()
    if Single:
        figsize=(figsize,figsize)
        fig, axes=plt.subplots(nrows= 1, ncols= 1,figsize=figsize)

    else:
        figsize=(figsize*2,figsize)
        fig, axes=plt.subplots(nrows= 1, ncols= 2,figsize=figsize)

    tick_params = {'labelsize': font_size}
    if not Single:
        output=rho.squeeze()
        ax=axes[0]
        output=output[ind,:]
        if scaling=="Linf":
            output=output/torch.max(torch.abs(cat2complex(output)))
        elif scaling=="L1":
            output=output/torch.sum(torch.abs(cat2complex(output)))
        img=torch.abs(cat2complex(output.squeeze())).view(xpix, ypix)
        pcol1=ax.pcolor(img.detach().cpu(),cmap='jet')
        cbar_used=plt.colorbar(pcol1,ax=ax)

    #plt.title('true', fontsize=font_size)
        cbar_used.ax.tick_params(labelsize=font_size)
        ax.tick_params(axis='both', **tick_params)
        #if scaling=="Linf":
        cbar_used.remove()
            


    output=rho_hat.squeeze() 
    output=output[ind,:]
    if scaling=="Linf":

        output=output/torch.max(torch.abs((output)))
    elif scaling=="L1":
        output=output/torch.sum(torch.abs((output)))
    img=torch.abs((output.squeeze())).view(xpix, ypix)
    if not Single:
        ax=axes[1]
    else:
        ax=plt.gca()
    pcol2=ax.pcolor(img.detach().cpu(),cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
#plt.title('true', fontsize=font_size)
    #ax.set_title('Full model', fontsize=font_size)
    if not Single:  
        cbar.mappable.set_clim(*cbar_used.mappable.get_clim())
    elif Lclim!=None:
        cbar.mappable.set_clim(Lclim, Hclim)
        cbar.remove()
    cbar.ax.tick_params(labelsize=font_size)
    #cbar.remove()
    ax.tick_params(axis='both', **tick_params)
    plt.show()

def plot_2_imgs_XY(rho, rho_hat,figsize=8,font_size=50,xpix=31, ypix=21, file_name=None):
    plt.close()
    figsize=(figsize*2,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 2,figsize=figsize)

    tick_params = {'labelsize': font_size}
    output=rho.squeeze()
    ax=axes[0]
    true_x=rho[:,0].cpu().detach().numpy()
    true_y=rho[:,1].cpu().detach().numpy()


    #plt.title('true', fontsize=font_size)
    ax.tick_params(axis='both', **tick_params)
    ax.scatter(true_x, true_y)
    x_pred=rho_hat[:,0].cpu().detach().numpy()
    y_pred=rho_hat[:,1].cpu().detach().numpy()
            

    ax=axes[1]
    ax.scatter(x_pred, y_pred)
    ax.tick_params(axis='both', **tick_params)
    plt.show()


def plot_2_imgs_XY_displacments(rho, rho_hat,figsize=8,font_size=50,xpix=31, ypix=21, file_name=None):
    plt.close()
    figsize=(figsize,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 1,figsize=figsize)
    ax=plt.gca()
    tick_params = {'labelsize': font_size}
    output=rho.squeeze()
    true_x=rho[:,0].cpu().detach().numpy()
    true_y=rho[:,1].cpu().detach().numpy()


    #plt.title('true', fontsize=font_size)
    ax.tick_params(axis='both', **tick_params)
    x_pred=rho_hat[:,0].cpu().detach().numpy()
    y_pred=rho_hat[:,1].cpu().detach().numpy()
    plt.quiver(x_pred, y_pred, true_x-x_pred, true_y-y_pred,label='Displacement',angles='xy', scale_units='xy', scale=1)
    ax.scatter(true_x, true_y, label='True')
    ax.scatter(x_pred, y_pred, label='Predicted')
    ax.tick_params(axis='both', **tick_params)
    plt.legend()
    plt.show()





#plots km image given a single config and sensing matrix
def KM_img(rho, sensing,figsize=8,scaling='Linf',font_size=50, file_name=None, xpix=31, ypix=21,WAND=False):
    plt.close()

    b=sensing@rho.T
    img=sensing.T.conj()@b
    
    figsize=(figsize,figsize)
    fig=plt.figure(figsize=figsize)

    tick_params = {'labelsize': font_size}



    output=img

    output=np.abs(output)/np.max(np.abs((output)))
    
    output=output.reshape(xpix, ypix)

    ax=plt.gca()
    pcol2=ax.pcolor(output,cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
    
    cbar.ax.tick_params(labelsize=font_size)
    #cbar.remove()
    ax.tick_params(axis='both', **tick_params)
    if file_name!=None:
        plt.savefig(f'/home/achristie/Codes_data/E_D_figs/{file_name}.pdf')
    plt.tight_layout()
    if WAND:
        
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        
        del fig, buf
        return [wandb.Image(img)]
    else:
        plt.show()



#for plotting permed data
def raw_img_perm(rho, perm1,perm2,font_size=50, file_name=None, xpix=31, ypix=21,WAND=False,figsize=8):
    plt.close()
    img=np.linalg.inv(perm1)@rho.T
    
    figsize=(figsize,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 2,figsize=figsize)
    ax=axes[0]
    tick_params = {'labelsize': font_size}
    output=img
    output=np.abs(output)/np.max(np.abs((output)))
    output=output.reshape(xpix, ypix)
    pcol2=ax.pcolor(output,cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.remove()
    ax.tick_params(axis='both', **tick_params)

    img=np.linalg.inv(perm2)@rho.T
    ax=axes[1]
    tick_params = {'labelsize': font_size}
    output=img
    output=np.abs(output)/np.max(np.abs((output)))
    output=output.reshape(xpix, ypix)
    pcol2=ax.pcolor(output,cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.remove()
    ax.tick_params(axis='both', **tick_params)
    if file_name!=None:
        plt.savefig(f'/home/achristie/Codes_data/E_D_figs/{file_name}.pdf')
    plt.tight_layout()
    if WAND:
        
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        
        del fig, buf
        return [wandb.Image(img)]
    else:
        plt.show()


#plots raw targets
def raw_img(rho, font_size=50, file_name=None, xpix=31, ypix=21,WAND=False,figsize=8):
    plt.close()
    
    figsize=(figsize,figsize)
    figs=plt.subplots(figsize=figsize)
    ax=plt.gca()
    tick_params = {'labelsize': font_size}
    output=rho
    output=np.abs(output)/np.max(np.abs((output)))
    output=output.reshape(xpix, ypix)
    pcol2=ax.pcolor(output,cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.remove()
    ax.tick_params(axis='both', **tick_params)
    if file_name!=None:
        plt.savefig(f'/home/achristie/Codes_data/E_D_figs/{file_name}.pdf')
    plt.tight_layout()
    if WAND:
        
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        
        del fig, buf
        return [wandb.Image(img)]
    else:
        plt.show()

#generates data using sensing matrix in locat 
def Generate_data_pnas(locat, amount, S=4, seed=0, pixels='One-hot'):
    np.random.seed(seed)    
    try:
        medium= np.array(mat73.loadmat(locat+'/rtt.mat')['Artt'])

    except:
        print('No medium found')

    data_rho=np.zeros((amount,medium.shape[-1]))
    if pixels=='Gaussian_abs':
        data_rho=np.zeros((amount,medium.shape[-1]))+1j*np.zeros((amount,medium.shape[-1]))

    #data_b=np.zeros((amount,631))
    for i in range(amount):
        if pixels=='One-hot':
            data_rho[i][:S]=1
        elif pixels=='soft':
            data_rho[i][:S]=1/S
        elif pixels=='Gaussian':
            data_rho[i][:S]=(.1+np.abs(np.random.randn(S)))*np.sign(np.random.randn(S))
            #data_rho[i]=abs(data_rho[i])/sum(abs(data_rho[i]))
        elif pixels=='Gaussian_abs':
            data_rho[i][:S]=(np.random.randn(S))+1j*(np.random.randn(S))
            data_rho[i][:S]=data_rho[i][:S]/abs(data_rho[i][:S])
            #sign=np.sign(np.random.randn(data_rho[i][:S].shape))
            #data_rho[i][:S]=data_rho[i][:S]*sign
            #data_rho[i]=abs(data_rho[i])/sum(abs(data_rho[i]))
        perm = np.random.permutation(medium.shape[-1])
        data_rho[i]=data_rho[i][perm]
    data_b=medium@data_rho.T
    print(f"Medium: {medium.shape}, Rho: {data_rho.shape}, B: {data_b.T.shape}")

    return data_rho, data_b.T

def Generate_data_pnas_EYE(locat, medium='Random'):
    if medium=='Random':
        medium= np.array(mat73.loadmat(locat+'/rtt.mat')['Artt'])
    else:
        medium= np.array(mat73.loadmat(locat+'/G_0.mat')['A0'])
        
    data_rho=np.eye(medium.shape[-1])
    data_b=medium@data_rho.T
    print(f"Medium: {medium.shape}, Rho: {data_rho.shape}, B: {data_b.T.shape}")

    return data_rho, data_b.T

#Perms target then plots KM image
def MDS_wandbout_KM(perm, g_permed):
    g_permed=cat2complex(g_permed).reshape((310,100))
    perm=perm.reshape(100,100)
    g=g_permed@np.linalg.inv(perm)
    rho=np.zeros(100)
    rho[55]=1

    return KM_img(rho, g,xpix=10,ypix=10, font_size=25, WAND=True)
    #H.KM_img(rho, g_permed_single@np.linalg.inv(perm_single),xpix=10,ypix=10, font_size=25)

#Same as aboe but with target and predicted only perm.
def MDS_wandbout_raw(perm1, perm2,scat_locat=[0,1,2,3]):
    perm1=perm1.reshape(100,100)
    perm2=perm2.reshape(100,100)
    rho=np.zeros(100)
    rho[scat_locat]=1
    return raw_img_perm(rho, perm1, perm2, xpix=10,ypix=10, font_size=25, WAND=True)
    #H.KM_img(rho, g_permed_single@np.linalg.inv(perm_single),xpix=10,ypix=10, font_size=25)

#reformats sweep input for a single run
def reformat_sweep_for_1_run(param_dict):
    new_dict={}
    for key, value in param_dict.items():
        for val_key, val_val in value.items():
            new_dict[key]=val_val[0]
    return new_dict

    

def KL_divergence(mean, logvar):
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())


#The below was written by Mattheiu Blondel from 
#mblondel/projection_simplex_vectorized.py
def projection_simplex(V, z=1, axis=None):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()


def _projection_simplex(v, z=1):
    """
    Old implementation for test and benchmark purposes.
    The arguments v and z should be a vector and a scalar, respectively.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def plot_2_unordered_imgs(epoch, rho, rho_hat,ind=9,figsize=8,scaling='Linf',font_size=50, Single=False, xpix=31, ypix=21,xpix2=31,ypix2=21, file_name=None):
    plt.close()
    if Single:
        figsize=(figsize,figsize)
        fig, axes=plt.subplots(nrows= 1, ncols= 1,figsize=figsize)

    figsize=(figsize*2,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 2,figsize=figsize)

    tick_params = {'labelsize': font_size}
    output=rho.squeeze()
    ax=axes[0]
    output=output[ind,:]
#    if scaling=="Linf":
#        output=output/torch.max(torch.abs(cat2complex(output)))
#    elif scaling=="L1":
#        output=output/torch.sum(torch.abs(cat2complex(output)))
    img=torch.abs(cat2complex(output.squeeze())).view(xpix, ypix)
    pcol1=ax.pcolor(img.detach().cpu(),cmap='jet')
    cbar_used=plt.colorbar(pcol1,ax=ax)

    #plt.title('true', fontsize=font_size)
    cbar_used.ax.tick_params(labelsize=font_size)
    ax.tick_params(axis='both', **tick_params)
    #if scaling=="Linf":
    cbar_used.remove()
            


    output=rho_hat.squeeze() 
    output=output[ind,:]
    if scaling=="Linf":

        output=output/torch.max(torch.abs((output)))
    elif scaling=="L1":
        output=output/torch.sum(torch.abs((output)))
    img=torch.abs((output.squeeze())).view(xpix2, ypix2)
    if not Single:
        ax=axes[1]
    else:
        ax=plt.gca()
    pcol2=ax.pcolor(img.detach().cpu(),cmap='jet')
    cbar=plt.colorbar(pcol2, ax=ax)
    ax.set_title('Full model', fontsize=font_size)
    cbar.mappable.set_clim(*cbar_used.mappable.get_clim())
    
    cbar.ax.tick_params(labelsize=font_size)
    #cbar.remove()
    ax.tick_params(axis='both', **tick_params)
    fig = plt.gcf()
    ax = plt.gca()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
     
    del fig, buf
    return [wandb.Image(img, caption="Epoch "+str(epoch))]


def hist_2_wandb(data, figsize=8, font_size=50, epoch=0):
    #if min(data)<.50:
    counts, bins = np.histogram(data, bins=[.01*i for i in range(101)])
    #elif min(data)<.75:
    #    counts, bins = np.histogram(data, bins=[.01*i+.5 for i in range(51)])
    #else:
    #    counts, bins = np.histogram(data, bins=[.01*i+.75 for i in range(26)])
    
    plt.close()

    figsize=(figsize,figsize)
    fig, axes=plt.subplots(nrows= 1, ncols= 1,figsize=figsize)
    tick_params = {'labelsize': font_size}
    
    plt.title('Histogram max_j<ghat_i, g_j>')
    plt.hist(data, bins=bins)
    
    fig = plt.gcf()
    ax = plt.gca()
    ax.tick_params(axis='both', **tick_params)

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    
    del fig, buf
    return [wandb.Image(img, caption="Epoch "+str(epoch))]