import tensorflow as tf
import os
import numpy as np
import nibabel as nib

class data_frame:
    """
    A single character class.
    """
    def __init__(self, batch_dim, primary_axis):
        #this helps keep data handling consistent
        self.batch_dim = batch_dim
        self.primary_axis = primary_axis

        self.tf_dataset = None
        self.file_list = None


    def load_data(self,train_list,data_dir,load_labels=False,label_dir=None,**kwargs):
        if load_labels:
            print('loading labels, forcing shuffle OFF')
            kwargs['shuffle_order']=False

        file_list = get_file_list(train_list,data_dir)
        tf_dataset = create_dataset(file_list,self.batch_dim,self.primary_axis,**kwargs)

        if load_labels:
            #load data labels as well
            old_filenames = [fl_i.split('.')[0] for fl_i in train_list]
            new_filenames = [fl_i+'_label' for fl_i in old_filenames]
            file_list_l = [fl_i.replace(old_f,new_f) for fl_i,old_f,new_f in zip(train_list,old_filenames,new_filenames)]

            if label_dir == None:
                #assume labels are in the same directory
                new_data_dir = data_dir
            else:
                new_data_dir = label_dir
                #optionally add label to dir name
                #new_data_dir = data_dir[:-1] if data_dir[-1] == '/' else data_dir +'_label'

            file_list_l = get_file_list(file_list_l,new_data_dir)
            #load labels
            tf_dataset_l = create_dataset(file_list_l,self.batch_dim,self.primary_axis,loading_label=True,**kwargs)
            #pack data and labels together
            self.tf_dataset = tf.data.Dataset.zip((tf_dataset,tf_dataset_l))
            self.file_list = (file_list, file_list_l)
            return

        else:
            #just data
            self.tf_dataset = tf_dataset 
            self.file_list = file_list
            return


    def save_nii(self,out_array,results_dir,out_fname):

        #rearrange array to default orientation
        out_array = np.rollaxis(out_array,0,self.primary_axis+1)#REVERSE roll
        out_array = out_array[...,0]#REMOVE 'channel' dimension

        #out_array = out_array.astype(np.float16)
        out_nii = nib.Nifti1Image(out_array, affine=np.eye(4))
        nib.save(out_nii, os.path.join(results_dir,out_fname))

        return



def get_file_list(f_list_path,data_dir):
    if isinstance(f_list_path, str):
        #read file in path and extract file names 
        with open(f_list_path, 'r') as f:
            #for item in list_i:
            file_list = f.readlines()
    elif isinstance(f_list_path, list) :
        #alternatively accept list of file names
        file_list = f_list_path
    else:
        raise ValueError("Unexpected type for train_list, must be list or path to text file")


    file_list = [os.path.join(data_dir,f_i.split('\n')[0]) for f_i in file_list]

    return file_list


def preprocess_func(image):
    #for now, do nothing 
    return image



def _read_nib_func(img_path,exception_size,primary_axis,loading_label):
    img_path = str(img_path.numpy(), 'utf-8')#.numpy extracts the string from the tensor

    #try block handles case where label does not exist, returns zero-filled volume
    #if not explicitly loading a label, raise error as per normal
    try:
        img_file = nib.load(img_path)
        img_decoded = img_file.get_fdata() # access images
        img_decoded = img_decoded.astype(np.float32)#match tensor type expected
    except FileNotFoundError as e: #Exception as e:
        if loading_label:
            print('Assuming zero-filled data: '+str(e))
            img_decoded = np.zeros(exception_size).astype(np.float32)
        else:
            raise e

    #primary axis will be put first, 2 for brain, 1 for abdomen
    img_decoded = np.rollaxis(img_decoded,primary_axis)#change axes

    if len(np.shape(img_decoded)) <4:
        img_decoded = img_decoded[...,None]#add 'channel' dimension
    
    return img_decoded



def create_dataset(file_list,batch_dim,primary_axis,preproc_f = preprocess_func,shuffle_order=True,loading_label=False,normalize=True):
    #create dataset 
    data_from_list = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle_order:
        data_from_list = data_from_list.shuffle(len(file_list))
    #use non-tensorflow function to read nifti file, flat_map combines slices from volumes
    #size and label arguments used to return zero-filled volume for non-existent labels
    data_from_list = data_from_list.flat_map(lambda f_path:
             tf.data.Dataset.from_tensor_slices(tuple(
             tf.py_function(lambda x:_read_nib_func(x,
             batch_dim[:-1],#exception size default volume
             primary_axis,#axis in volume to put first
             loading_label),#if loading label give option of defaulting to zeros-filled volume             
             [f_path],[tf.float32]))))#inputs and outputs of py_func
    #data_from_list = data_from_list.map(_read_nib_func)

    #without preproc_f it seems to return volume instead of flattened
    data_from_list = data_from_list.map(preproc_f, num_parallel_calls=4)
    data_from_list = data_from_list.batch(batch_dim[0])
    data_from_list = data_from_list.prefetch(1)

    return data_from_list


