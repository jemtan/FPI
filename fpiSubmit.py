import numpy as np
import itertools
import copy 
from datetime import datetime
import os
import pickle

from sklearn.metrics import average_precision_score
import tensorflow as tf
import readData
import self_sup_task
from models.wide_residual_network import create_wide_residual_network_selfsup
from scipy.signal import savgol_filter
from utils import save_roc_pr_curve_data
import gc


def train_folder(input_dir,output_dir,mode,data):

    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

    data_frame = get_data_frame(data,input_dir,shuffle_order=True)
    mdl = get_mdl(data,data_frame,restore=False)

    submit_train(mdl,data_frame,output_dir,data)

    return

def predict_folder(input_dir,output_dir,mode,data):
    #K.manual_variable_initialization(True)
    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)

    data_frame = get_data_frame(data,input_dir,shuffle_order=False)
    mdl = get_mdl(data,data_frame,restore=True)

    submit_test(mdl,data_frame,output_dir,mode)

    return

def get_data_frame(data,input_dir,shuffle_order=False,load_labels=False):
    if 'brain' in data:
        batch_dim = [256,256,256,1]
        primary_axis = 2

    elif 'abdom' in data:
        batch_dim = [512,512,512,1]
        primary_axis = 1

    else:
        raise ValueError("data type not correctly defined. Either choose 'brain','abdom', or add a new definition")


    data_frame = readData.data_frame(batch_dim,primary_axis)
    input_list = os.listdir(input_dir)
    data_frame.load_data(input_list,input_dir,shuffle_order=shuffle_order,load_labels=load_labels)
    return data_frame

def get_mdl(data,data_frame,restore=False):

    if 'brain' in data:
        n, k = (16,4)#network size
        net_f='create_wide_residual_network_dec'
        n_classes = 1
        model_dir = '/workspace/restore_dir/brain/'

    elif 'abdom' in data:
        n, k = (19,4)#network size
        net_f='create_wide_residual_network_decdeeper'
        n_classes = 5
        model_dir = '/workspace/restore_dir/abdom/'

    else:
        raise ValueError("data type not correctly defined. Either choose 'brain','abdom', or add a new definition")


    if restore:
        #grab weights and build model
        model_fnames = os.listdir(model_dir)
        model_fnames = [fn for fn in model_fnames if 'weights' in fn][0]
        model_path = os.path.join(model_dir,model_fnames)
        print(model_path)
        mdl = tf.keras.models.load_model(model_path)

    else:
        #build new model
        mdl = create_wide_residual_network_selfsup(data_frame.batch_dim[1:],
            n_classes, n, k, net_f=net_f)

    return mdl

@tf.function
def train_step(mdl,x, y):
    loss_fn = mdl.compiled_loss
    with tf.GradientTape() as tape:
        logits = mdl(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, mdl.trainable_weights)
    mdl.optimizer.apply_gradients(zip(grads, mdl.trainable_weights))
    mdl.compiled_metrics.update_state(y, logits)
    return loss_value

@tf.function
def test_step(mdl,x, y):
    loss_fn = mdl.compiled_loss    
    logits = mdl(x, training=False)
    loss_value = loss_fn(y, logits)        
    return loss_value

@tf.function
def pred_step(mdl,x):
    pred = mdl(x, training=False)
    return pred

def grouped(iterable, n):
    #get n elements at a time
    return zip(*[iter(iterable)]*n)


def submit_train(mdl,data_frame,output_dir,data,epochs=50,cyclic_epochs=0,save_name='selfsup_mdl',training_batch_size=32):
    print('training start: {}'.format(datetime.now().strftime('%Y-%m-%d-%H%M')))

    num_classes = mdl.output_shape[-1]
    num_classes = None if num_classes <= 1 else num_classes
    fpi_args = {'num_classes':num_classes,
                'core_percent':0.5 if 'brain' in data else 0.8,
                'tolerance': None if 'brain' in data else 1E-3
    }

    elem_in_epoch = len(data_frame.file_list)
    
    if cyclic_epochs>0:
        half_cycle_len = elem_in_epoch//4
        lr_min = 1E-4
        lr_max = 1E-1
        half1 = np.linspace(lr_min,lr_max,half_cycle_len)
        half2 = np.linspace(lr_max,lr_min,half_cycle_len)
        lr_cycle = np.concatenate((half1,half2),0)

    for epoch_i in range(epochs+cyclic_epochs):
        if epoch_i>epochs and elem_i < len(lr_cycle):
            #cyclic training portion, adjust learning rate
            tf.keras.backend.set_value(mdl.optimizer.lr, lr_cycle[elem_i])

        #get subjects in pairs for mixing
        for batch_in,batch_in2 in grouped(data_frame.tf_dataset,2):
            #apply fpi on batch
            pex1,pex2 = self_sup_task.patch_ex(batch_in,batch_in2,**fpi_args)
            ind_sampler = index_sampling(len(pex1[0]))#randomize slices in batch
            for _ in range(len(pex1[0])//training_batch_size):
                cur_inds = ind_sampler.get_inds(training_batch_size)
                train_step(mdl,tf.gather(pex1[0],cur_inds),tf.gather(pex1[1],cur_inds))
                train_step(mdl,tf.gather(pex2[0],cur_inds),tf.gather(pex2[1],cur_inds))

        print('epoch {}: {}'.format(str(epoch_i),datetime.now().strftime('%Y-%m-%d-%H%M')))
    
        #measure loss
        for batch_in,batch_in2 in grouped(data_frame.tf_dataset,2):
            break
        pex1,pex2 = self_sup_task.patch_ex(batch_in,batch_in2,**fpi_args)
        avg_loss = []
        ind_sampler = index_sampling(len(pex1[0]))#randomize slices in batch
        for _ in range(len(pex1[0])//training_batch_size):
            cur_inds = ind_sampler.get_inds(training_batch_size)
            avg_loss.append(test_step(mdl,tf.gather(pex1[0],cur_inds),tf.gather(pex1[1],cur_inds)))
            avg_loss.append(test_step(mdl,tf.gather(pex2[0],cur_inds),tf.gather(pex2[1],cur_inds)))
        avg_loss = np.mean(avg_loss)
        print('Avg loss: {}'.format(avg_loss))
        if epoch_i == 0:
            best_loss = avg_loss
        elif avg_loss < best_loss:
            best_loss = avg_loss
            print('new best loss')
            save_model(mdl,output_dir,save_name+'_bestLoss',time_stamp=False)

        if epoch_i % 10 == 0 or epoch_i>epochs:
            #save every 10 epochs or every epoch in cyclic mode
            save_model(mdl,output_dir,save_name)
        
    #save final model
    save_model(mdl,output_dir,save_name+'_final')
    return


def submit_test(mdl,data_frame,output_dir,mode,batch_size=1,save_name='selfsup_mdl'):
    print('testing start: {}'.format(datetime.now().strftime('%Y-%m-%d-%H%M')))
    
    nii_file = 0
    for batch_in in data_frame.tf_dataset:

        #predict for subject
        pred = np.zeros(np.shape(batch_in))
        for ind in range(len(batch_in)//batch_size): 
            pred[ind:(ind+1)*batch_size] = pred_step(mdl,batch_in[ind:(ind+1)*batch_size])

        output_chan = np.shape(pred)[-1]
        if output_chan > 1:
            pred *= np.arange(output_chan)/(output_chan-1)
            pred = np.sum(pred,-1,keepdims=True)

        #save output as nifti and label with label suffix
        #print(data_frame.file_list[0])#only data, not label names
        fname_i = data_frame.file_list[nii_file].split('/')[-1]
             

        if 'sample' in mode: 
            #derive subject-level score
            im_level_score = np.mean(pred,axis=(1,2,3))
            window_size = int((len(im_level_score)*0.1)//2)*2+1#take 10% sliding filter window
            im_level_score_f = savgol_filter(im_level_score,window_size,3)#order 3 polynomial

            im_level_score_s = sorted(im_level_score_f)
            im_level_score_s = im_level_score_s[int(len(im_level_score_s)*0.75):]
            sample_score = np.mean(im_level_score_s)#mean of top quartile values
            with open(os.path.join(output_dir,fname_i + ".txt"), "w") as write_file:
                write_file.write(str(sample_score))

        if 'pixel' in mode:
            data_frame.save_nii(pred,output_dir,fname_i)

        nii_file += 1

    return


def save_model(mdl,results_dir,fname,time_stamp=True):
    #save model
    if time_stamp:
        #mdl_weights_name = fname+'_{}_weights.h5'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
        mdl_weights_name = fname+'_{}_weights'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    else:
        #mdl_weights_name = fname+'_weights.h5'
        mdl_weights_name = fname+'_weights'

    mdl_weights_path = os.path.join(results_dir, mdl_weights_name)
    mdl.save(mdl_weights_path)

    return



class index_sampling(object):
    def __init__(self,total_len):
        self.total_len = total_len
        self.ind_generator = rand_ind_fisheryates(self.total_len)

    def get_inds(self,batch_size):
        cur_inds = list(itertools.islice(self.ind_generator,batch_size))
        if len(cur_inds) < batch_size:
            #end of iterator - reset/shuffle
            self.ind_generator = rand_ind_fisheryates(self.total_len)
            cur_inds = list(itertools.islice(self.ind_generator,batch_size))
        return cur_inds

    def reset():
        self.ind_generator = rand_ind_fisheryates(self.total_len)
        return    

def rand_ind_fisheryates(num_inds):
    numbers=np.arange(num_inds,dtype=np.uint32)
    for ind_i in range(num_inds):
        j=np.random.randint(ind_i,num_inds)
        numbers[ind_i],numbers[j]=numbers[j],numbers[ind_i]
        yield numbers[ind_i]



