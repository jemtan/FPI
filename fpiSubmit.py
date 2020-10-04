from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import numpy as np
import itertools
import var_ops as vops
import copy 
from datetime import datetime
import os
import pickle

from sklearn.metrics import average_precision_score
from scipy.signal import savgol_filter
import nibabel as nib
import tensorflow as tf
import readData
import self_sup_task

from models.wide_residual_network import create_wide_residual_network_selfsup
from keras import losses,optimizers



def train_folder(input_dir,output_dir,mode,data):
    sess = K.get_session()

    data_frame = get_data_frame(data,input_dir,shuffle_order=True)
    mdl = get_mdl(data,data_frame,restore=False)

    submit_train(sess,mdl,data_frame,output_dir,data)

    return

def predict_folder(input_dir,output_dir,mode,data):
    sess = K.get_session()
    K.manual_variable_initialization(True)

    data_frame = get_data_frame(data,input_dir)
    mdl = get_mdl(data,data_frame)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #K.set_session(sess)
       
    submit_test(sess,mdl,data_frame,output_dir,mode)

    #K.clear_session()
    return

def get_data_frame(data,input_dir,shuffle_order=False):
    if 'brain' in data:
        vol_dim = 256
        primary_axis = 2

    elif 'abdom' in data:
        vol_dim = vol_dim = 512
        primary_axis = 1  
    else:
        raise ValueError("data type not correctly defined. Either choose 'brain' or 'abdom'")


    data_frame = readData.data_frame(vol_dim,primary_axis)
    input_list = os.listdir(input_dir)
    data_frame.load_mood(input_list,input_dir,shuffle_order=shuffle_order)
    return data_frame

def get_mdl(data,data_frame,restore=True):

    batch_shape = (data_frame.vol_dim, data_frame.vol_dim, 1)
    if 'brain' in data:
        n, k = (16,4)#network size
        net_f='create_wide_residual_network_dec'
        n_classes = 5
        model_dir = '/workspace/restore_dir/brain/'
        
    elif 'abdom' in data:
        n, k = (19,4)#network size
        net_f='create_wide_residual_network_decdeeper'
        n_classes = 5
        model_dir = '/workspace/restore_dir/abdom/'
    else:
        raise ValueError("data type not correctly defined. Either choose 'brain' or 'abdom'")

    if n_classes == 1:
        activation = 'sigmoid'
        loss_f = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss_f = 'categorical_crossentropy'


    mdl = create_wide_residual_network_selfsup(batch_shape,n_classes, n, k,
        net_f=net_f,final_activation=activation)
    adam_reptile = optimizers.Adam(lr=0.001)
    mdl.compile(adam_reptile,[loss_f],['acc'])

    if restore:
        model_fnames = os.listdir(model_dir)
        model_fnames = [fn for fn in model_fnames if 'opt' not in fn][0]
        model_path = os.path.join(model_dir,model_fnames)
        f_w = model_path
        f_o = model_path.replace('_weights.h5','_opt_weights.pkl')
        print([f_w,f_o])
        mdl.load_weights(f_w)
        #restore optimizer
        #with open(f_o, 'rb') as fpickle:
        #    weight_values = pickle.load(fpickle)
        #K.set_value(mdl.optimizer.iterations, 0)#manual init
        #mdl.optimizer.set_weights(weight_values)#TODO: Check whether same optimizer whether to restore or not

    else:
        mdl._make_train_function()


    return mdl

def submit_train(sess,mdl,data_frame,output_dir,data,batch_size=8,epochs=50,cyclic_epochs=10):
    print('training start: {}'.format(datetime.now().strftime('%Y-%m-%d-%H%M')))
   
    num_classes = mdl.output_shape[-1]
    num_classes = None if num_classes <= 1 else num_classes
    elem_in_epoch = len(data_frame.file_list)
    core_percent = 0.5 if 'brain' in data else 0.8 #0.5 for brain, 0.8 for abdomen
    tol = None if 'brain' in data else 1E-3
    
    if cyclic_epochs>0:
        half_cycle_len = elem_in_epoch//4
        lr_min = 1E-4
        lr_max = 1E-1
        half1 = np.linspace(lr_min,lr_max,half_cycle_len)
        half2 = np.linspace(lr_max,lr_min,half_cycle_len)
        lr_cycle = np.concatenate((half1,half2),0)

    for epoch_i in range(epochs+cyclic_epochs):
        sess.run(data_frame.init_op)#reinitialization with shuffle 

        for elem_i in range(elem_in_epoch//2):#read 2 elems each time
            if epoch_i>epochs and elem_i < len(lr_cycle):
                #cyclic training portion, adjust learning rate
                K.set_value(mdl.optimizer.lr, lr_cycle[elem_i])

            #each elem should be one patient volume
            elem1 = sess.run(data_frame.next_elem)
            elem2 = sess.run(data_frame.next_elem)

            elem_ex1,elem_ex2,elem_label = self_sup_task.patch_ex(elem1,elem2,num_classes,core_percent=core_percent,tolerance=tol)
            ind_sampler = index_sampling(len(elem_ex1))#randomize slices in batch

            for _ in range(len(elem_ex1)//batch_size):
                cur_inds = ind_sampler.get_inds(batch_size)
                mdl.train_on_batch(elem_ex1[cur_inds],elem_label[cur_inds])
                mdl.train_on_batch(elem_ex2[cur_inds],elem_label[cur_inds])

        print('epoch {}: {}'.format(str(epoch_i),datetime.now().strftime('%Y-%m-%d-%H%M')))
        #measure loss
        sess.run(data_frame.init_op)
        elem1 = sess.run(data_frame.next_elem)
        elem2 = sess.run(data_frame.next_elem)
        elem_ex1,elem_ex2,elem_label = self_sup_task.patch_ex(elem1,elem2,num_classes,core_percent=core_percent,tolerance=tol)#exchange patches
        ind_sampler = index_sampling(len(elem_ex1))

        avg_loss = []
        for _ in range(len(elem_ex1)//batch_size):
            cur_inds = ind_sampler.get_inds(batch_size)
            avg_loss.append(mdl.test_on_batch(elem_ex1[cur_inds],elem_label[cur_inds])[0])#only loss
        avg_loss = np.mean(avg_loss)
        print('Avg loss: {}'.format(avg_loss))
        if epoch_i == 0:
            best_loss = avg_loss
        elif avg_loss < best_loss:
            best_loss = avg_loss
            print('new best loss')
            save_model(mdl,output_dir,'selfsup_mdl_bestLoss',time_stamp=False)

        if epoch_i % 10 == 0 or epoch_i>epochs:
            #save every 10 epochs or every epoch in cyclic mode
            save_model(mdl,output_dir,'selfsup_mdl')


    #save final model
    save_model(mdl,output_dir,'selfsup_mdl_final')
    return


def submit_test(sess,mdl,data_frame,output_dir,mode,batch_size=1):
    print('testing start: {}'.format(datetime.now().strftime('%Y-%m-%d-%H%M')))
    
    sess.run(data_frame.init_op)#data
    
    subject_i = 0
    data_remaining = True
    while data_remaining:
        try:#keep going for all data in iterator
            batch = sess.run(data_frame.next_elem)
 
            #for batch_i,label_i in zip(batch,label):
            pred = mdl.predict(batch,batch_size)
            output_chan = np.shape(pred)[-1]
            if output_chan > 1:
                pred *= np.arange(output_chan)/(output_chan-1)
                pred = np.sum(pred,-1,keepdims=True)
            #save output as nifti and label with label suffix
            #print(data_frame.file_list[0])#only data, not label names
            fname_i = data_frame.file_list[subject_i].split('/')[-1]
            if 'pixel' in mode:
                data_frame.save_nii(pred,output_dir,fname_i)

            elif 'sample' in mode: 
                #derive subject-level score
                im_level_score = np.mean(pred,axis=(1,2,3))
                window_size = int((len(im_level_score)*0.1)//2)*2+1#take 10% sliding filter window
                im_level_score_f = savgol_filter(im_level_score,window_size,3)#order 3 polynomial

                im_level_score_s = sorted(im_level_score_f)
                im_level_score_s = im_level_score_s[int(len(im_level_score_s)*0.75):]
                sample_score = np.mean(im_level_score_s)#mean of top quartile values
                
                with open(os.path.join(output_dir,fname_i + ".txt"), "w") as write_file:
                    write_file.write(str(sample_score))


            subject_i += 1

        except tf.errors.OutOfRangeError:
            print('end of partition')
            break

    return 




def save_model(mdl,results_dir,fname,time_stamp=True):
    #save model
    if time_stamp:
        mdl_weights_name = fname+'_{}_weights.h5'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    else:
        mdl_weights_name = fname+'_weights.h5'

    mdl_weights_path = os.path.join(results_dir, mdl_weights_name)
    mdl.save_weights(mdl_weights_path)

    #save optimizer
    if time_stamp:
        mdl_opt_weights_name = fname+'_{}_opt_weights.pkl'.format(datetime.now().strftime('%Y-%m-%d-%H%M'))
    else:
        mdl_opt_weights_name = fname+'_opt_weights.pkl'

    mdl_opt_weights_path = os.path.join(results_dir, mdl_opt_weights_name)
    symbolic_weights = getattr(mdl.optimizer,'weights')
    opt_weight_values = K.batch_get_value(symbolic_weights)
    with open(mdl_opt_weights_path,'wb') as fpickle:
        pickle.dump(opt_weight_values,fpickle)

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


