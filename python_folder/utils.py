from random import randint;
import numpy as np
import random
from random import randint
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import drawSvg as draw
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

# To save a trained model ! 
import torch.optim as optim

## Let's define all the functions needed 

def generate_matrix():
    # Generate a matrix (corresponding to a smiley) unconditionnally 
    matrix = np.zeros((10))
    
    #eye: 
    e_y = randint(0,50)
    e_x = randint(0,50)
    
    #mouth: 
    m1_x = randint(-50, -1)
    m1_y = randint(-50, e_y)
    m2_x = 0
    m2_y = randint(-50, e_y)
    m3_x = - m1_x
    m3_y = m1_y 
    
    matrix[:] = [-e_x, e_y, e_x, e_y, m1_x, m1_y, m2_x, m2_y, m3_x, m3_y]
    return matrix 


def generate_matrix_cond(y):
    # Generate a conditional matrix (corresponding to a smiley):
    # If y = 0 we generate a sad smiley
    # If y = 1 we generate a happy smiley
    matrix = np.zeros((10))
    
    #eye: 
    e_y = randint(0,50)
    e_x = randint(0,50)
    
    #mouth: 
    m1_x = randint(-50, -1)
    m1_y = randint(-50, e_y)
    
    if(y==0): #ie sad mouth
        m2_x = 0
        m2_y = randint(m1_y, e_y)
        
    if(y==1): #ie happy mouth
        m2_x = 0
        m2_y = randint(-50, m1_y)
    
    m3_x = - m1_x
    m3_y = m1_y 
    
    matrix[:] = [-e_x, e_y, e_x, e_y, m1_x, m1_y, m2_x, m2_y, m3_x, m3_y]
    return matrix 


def draw_smiley(matrix):
    # Draw a smiley from a matrix 
    
    [e1_x, e1_y, e2_x, e2_y, m1_x, m1_y, m2_x, m2_y, m3_x, m3_y] = matrix[:]
    
    ### Canvas: 
    d = draw.Drawing(100, 100, origin='center', displayInline=False)
    
    ### eye1: 
    d.append(draw.Circle(e1_x, e1_y, 1, stroke='black', stroke_width = 2, fill = 'none'))
    
    ### eye2: 
    d.append(draw.Circle(e2_x, e2_y, 1, stroke='black', stroke_width = 2, fill = 'none'))
    
    ### mouth: 
    p = draw.Path(stroke_width=2, stroke='black',
              fill='black', fill_opacity=0)
    p.M(m1_x, m1_y)  # Start path at point (b1_x, b1_y)
    p.Q(m2_x, m2_y, m3_x, m3_y) # 2nd point at (b2_x, b2_y) and final point at (b3_x, b3_y)
    
    d.append(p)
    
    return d



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience number (here = 10)."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu' # needed for plot_latent below


def plot_latent_uncond(model, data, num_batches=10):
    for i, x in enumerate(data):  # "i" is the the iteration number (in the loop)
        z = model.encoder(x.float().to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1]) #, c=y, cmap='tab10')  
        if i > num_batches:
            plt.colorbar()
            break


def plot_latent_cond(model, data, num_batches=10):
    # A simple function to visualize our 2D latent space
    for i, (x, y) in enumerate(data):  # "i" is the the iteration number (in the loop)
        z = model.encoder(x.float().to(device),y.float().to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1])  
        if i > num_batches:
            plt.colorbar()
            break




def draw_column(smileys):
    # function to display all the smileys in a column, without limitation in the number of smileys
    length = len(smileys) # return the number of smiley we want to draw 
    #Canvas :
    d = draw.Drawing(100, 100 * length, origin=(-50,-(100*length)+50), diplayInline = False)
    
    for i in range(0,length):
        
        [e1_x, e1_y, e2_x, e2_y, m1_x, m1_y, m2_x, m2_y, m3_x, m3_y] = smileys[i,:]
        
        e1_y -= i*100
        e2_y -= i*100
        m1_y -= i*100
        m2_y -= i*100
        m3_y -= i*100
        
        ### eye1: 
        d.append(draw.Circle(e1_x, e1_y, 1, stroke='black', stroke_width = 2, fill = 'none'))
    
        ### eye2: 
        d.append(draw.Circle(e2_x, e2_y, 1, stroke='black', stroke_width = 2, fill = 'none'))
    
        ### mouth: 
        p = draw.Path(stroke_width=2, stroke='black',
              fill='black', fill_opacity=0)
        p.M(m1_x, m1_y)  # Start path at point (b1_x, b1_y)
        p.Q(m2_x, m2_y, m3_x, m3_y) # 2nd point at (b2_x, b2_y) and final point at (b3_x, b3_y)
    
        d.append(p)
    return d 


# New fonction to displan in array

def draw_grid(smileys, size=10):
    #Here we consider that smileys is of size 100 to display a canvas of 10x10 smileys but we can change it by 
    # modifying the integer "size" (by default size = 10 )
    
    #Canvas :
    d = draw.Drawing(100*size, 100*size, origin=(-50,-(100*size)+50), diplayInline = False)
    
    for j in range(0,size):
        for i in range(0,size):
        
            [e1_x, e1_y, e2_x, e2_y, m1_x, m1_y, m2_x, m2_y, m3_x, m3_y] = smileys[j*size + i,:]

            e1_x += j*100
            e2_x += j*100
            m1_x += j*100
            m2_x += j*100
            m3_x += j*100
                     
            e1_y -= i*100
            e2_y -= i*100
            m1_y -= i*100
            m2_y -= i*100
            m3_y -= i*100
        
            ### eye1: 
            d.append(draw.Circle(e1_x, e1_y, 1, stroke='black', stroke_width = 2, fill = 'none'))
    
            ### eye2: 
            d.append(draw.Circle(e2_x, e2_y, 1, stroke='black', stroke_width = 2, fill = 'none'))
    
            ### mouth: 
            p = draw.Path(stroke_width=2, stroke='black',
              fill='black', fill_opacity=0)
            p.M(m1_x, m1_y)  
            p.Q(m2_x, m2_y, m3_x, m3_y)
    
            d.append(p)
        
    # separate each smiley with lines : 
    for i in range(0,size-1):
        d.append(draw.Lines(-50, -50 - 100*i, 950, -50-100*i, stroke='black')) #e_x1, e_y1, delta_x, delta_y
        d.append(draw.Lines(50+100*i, 50, 50+100*i, -950, stroke='black'))    #e_x1, e_y1, delta_x, delta_y         
        
    return d 