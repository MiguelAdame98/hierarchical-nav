from difflib import SequenceMatcher
import os
import torch
from torchvision.transforms import Compose
import glob
import numpy as np
from itertools import product
from dommel_library.datasets.dataset_factory import dataset_factory
from dommel_library.datasets import MemoryPool
from dommel_library.datasets.transforms import (
    Resize, ChannelFirst, RescaleShift, ToFloat,
    Squeeze, Subsample, Unsqueeze, Pad, Crop)
from dommel_library.datastructs import Dict, TensorDict, cat
from dommel_library.nn import module_factory
from dommel_library.nn.summary import summary
from dommel_library.train import (
    Trainer,
    loss_factory,
    optimizer_factory,
)
from dommel_library.modules.visualize import vis_images
from dommel_library.distributions.multivariate_normal import MultivariateNormal
from .models import GQNModel
from .train import ModelTrainer

# from test_benchmark import Benchmark

def get_model_parameters(log_dir, epoch=None):
    #check if we inputed a param file
    dir, ext = os.path.splitext(log_dir)
    if ext == '.pt':
        return log_dir

    model_dir = os.path.join(log_dir, "models")
    if epoch is None:
        model_files = glob.glob(os.path.join(model_dir, "*.pt"))
        model = max(model_files, key=os.path.getctime)
    else:
        model = os.path.join(model_dir, "model-{:04d}.pt".format(epoch))
    return model


class ToDist():
    """ Change tensor type to MultiVariateNormal """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                
                seq_dist = MultivariateNormal(sequence[key][:,:int(sequence[key].shape[-1] / 2)],sequence[key][:,int(sequence[key].shape[-1] / 2):])
                
                sequence[key] = seq_dist
        return sequence

def create_datasets(config, train=True):
    #
    # We wrap the file pools in a DictPool to cache the dataset
    # after transforms in memory, if the backend is not file.
    #
    # When sequence length is 1, we squeeze the time dimension
    #
    # We fetch sequences of double the length and subsample a timestep
    train_set = None
    transforms = [ToFloat()]

    if 'image' in config.dataset['keys']:
        try:
            size_cam = config.model.OZ.observations.image
        except AttributeError:
            if 'OZ' in config:
                size_cam = config.OZ.model.observations.image
            else:
                try:
                    size_cam = config.model.observations.image
                except AttributeError:
                    size_cam= [3,64,64]
        
        transforms.append(RescaleShift(1.0 / 255, 0, keys=["image"]))
        transforms.append(ChannelFirst(keys=["image"]))
        transforms.append(Resize(keys=["image"], size=size_cam))

    if 'posterior' in config.dataset['keys']:
        transforms.append(ToDist(keys=['posterior']))

    train_destination = config.dataset.get('train_destination',None)
    val_destination = config.dataset.get('val_destination',None)
        
    
    if train == True:
        # Wrap data in DictPool after transforming chunks of 300
        train_set = dataset_factory(config.dataset.train_set_location,
                                    train_destination,
                                    type=config.dataset.type,
                                    keys=config.dataset["keys"],
                                    sequence_length=config.dataset.sequence_length,
                                    sequence_stride=config.dataset.sequence_stride,
                                    transform=Compose(transforms), 
                                    cutoff = True)

        print('len train set', len(train_set))
    if config.dataset.val_set_location:
        val_set = dataset_factory(config.dataset.val_set_location,
                                  val_destination,
                                  type=config.dataset.type,
                                  keys=config.dataset["keys"],
                                  sequence_length=config.dataset.sequence_length,
                                  sequence_stride= config.dataset.sequence_stride,
                                  transform=Compose(transforms),
                                  cutoff = True)
        print('len val set', len(val_set))
        print('val set 0 shape', val_set[0].shape)

    else:
        val_set = None
    
    if train:
        if config.dataset.sequence_length == 1:
            # squeeze time dimension if sequence_length is 1
            train_set = MemoryPool(
                device=config.dataset.device,
                sequence_length=config.dataset.sequence_length,
                transform=Squeeze(keys=["image"], dim=0),).wrap(train_set)
            print('len train set', len(train_set))
            if val_set:
                val_set = MemoryPool(
                    device=config.dataset.device,
                    sequence_length=config.dataset.sequence_length,
                    transform=Squeeze(keys=["image"], dim=0)).wrap(val_set)
                print('len val set', len(val_set))
        else:
            train_set = MemoryPool(
                device=config.dataset.device).wrap(train_set)

            if val_set:
                val_set = MemoryPool(
                    device=config.dataset.device).wrap(val_set)
    
    return train_set, val_set


def create_model(config, train=True):
    # Construct model
    model_config = config.model
    model_type = model_config.get("type", None)
    if model_type == 'GQN':
        model = GQNModel(device = config.device, **model_config)
   
    else:
        #default option
        model = module_factory(**model_config)
        
    if not train:
        if "model_epoch" in config.keys():
            epoch = int(config.model_epoch)
        else:
            epoch = None
        if "params_dir" in model_config.keys():
            path = os.path.join(config.experiment_dir, model_config.params_dir)
            model.load(get_model_parameters(path, epoch = epoch))
        else:
            print('config log dir',config.log_dir)
            print('epoch,', epoch)
            try:
                model.load(get_model_parameters(config.log_dir, epoch= epoch), map_location='cpu')
            except TypeError:
                model.load(get_model_parameters(config.log_dir, epoch= epoch))
    
    model.to(config.device)
    #print(model)
    
    model.summary()


    return model 

def run(config):
    print("Loading data ...")
    train_set, val_set = create_datasets(config)

    print("Initializing model ...")
    model = create_model(config)

    loss = loss_factory(**config.loss)
    optimizer = optimizer_factory(loss.parameters() +
                                list(model.parameters()),
                                **config.optimizer)
    trainer = ModelTrainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        optimizer=optimizer,
        loss=loss,
        log_dir=config.log_dir,
        device=config.device,
        logging_platform="wandb",  # -> select the wandb platform
        experiment_config=config,  # -> store config in wandb as well
        ** config.trainer
    )

    # load from checkpoint
    print("Start training...")
    trainer.train(config.trainer.num_epochs, start_epoch=config.start_epoch)



import matplotlib.pyplot as plt

def evaluate(config):
    '''
    Call the benchmark to parse data, compute error and save data
    The rest is dependant on the model type

    n_sample (int): batch_size
    lookahead (int): predicted steps
    lookahead_ratio (bool): if True lookahead is interpreted as a %
    save_img_seq (bool): should we save the img pred/gt sequence
    unique_csv (bool): should the results of consecutive evaluation be saved in 1 file
    show_img (bool): do we want to display the evaluation img seq
    scenario (str): path of the scenario to run
    run_test (int): which test to run 
    1: run model with data than collect prediction and compare them -mse- to groundtruth
    '''
    #params for testbench, can be given as arguments to the main
    n_sample = config.get('n_sample', 3)
    lookahead = config.get('lookahead', 5)
    lookahead_ratio = config.get('lookahead_ratio', False)
    save_img_seq = config.get('save_img_seq', True)
    unique_csv = config.get('unique_csv', False)
    show_img = config.get('show_img', False)
    scenario = config.get('scenario', None)
    learn_steps = config.get('learn_steps', None)
    

    run_test = config.get('run_test', 1)

    

    #-- Create Model --
    gqn_oz_test_type = False #this is for the GQN trained with set Ozan model, currently we only have that (20/10/22)
    model_config = config.model.get("type", None)
    model = create_model(config, False)
    models =[model]

     #Check if we have Ozan model not in the trained model but as complemntary
    if ('OZ' in config.keys() or 'OZ' in config.model.keys()) and config.model['type']=='GQN':
        if 'OZ' in config.keys():
            #LOGDIR for PARAMS MUST BE SET IN OZ
            oz_model = create_model(config.OZ, False)
        elif 'OZ' in config.model.keys():
            oz_model = create_model(config.model.OZ, False)
        models.append(oz_model)
        gqn_oz_test_type = True


    #-- Collect Dataset --
    if 'scenario' in config.dataset and scenario== None:
        config.dataset.val_set_location = config.dataset.scenario
    elif scenario != None:
        config.dataset.val_set_location = scenario
    
    config.dataset.sequence_length = -1
    config.dataset.sequence_stride = -1
    _, eval_set = create_datasets(config, False) 


    test_setup = Benchmark(model_id = config.log_dir.split('/')[-1], data_file= config.dataset.val_set_location)
    model_set_data, pred_data = test_setup.parse_data(eval_set, n_sample=n_sample, learn_steps = learn_steps, lookahead=lookahead, lookahead_ratio=lookahead_ratio)
    
    if run_test ==1:
        
        #Run model from forward
        #m_output_seq, _ = run_model_step_by_step(models, model_set_data)
        #Run model from call
        m_output, models = run_model(models, model_set_data)
        

        #run the correct model type prediction process
        if model_config == 'GQN_OZ' or gqn_oz_test_type == True:
            pred_output = run_GQN_OZ_prediction_model(m_output, models, pred_data)
        elif model_config == 'GQN':
            pred_output = run_GQN_prediction_model(m_output, models, pred_data)
        elif model_config == 'Conv' :
            pred_output = run_OZ_prediction_model(models, pred_data)
        else :
            print('model %s not recognised', model_config)
    
        #Compute error in the benchmark and save data in a csv file
        keys = []
        if 'image_reconstructed' in pred_output:
            keys.append('image')
        if 'pose_reconstructed' in pred_output:
            keys.append('pose')

        mse_dict= test_setup.compute_mse_error(pred_output, key= keys, show_img = show_img)
        
        
        #Compute surprise between 2 steps
        # kl_dict = test_setup.compute_KL(state_dist_seq = m_output_seq['posterior'][1:], pref_dist_seq = m_output_seq['posterior'][:-1])
        # mse_dict.update(kl_dict)
        test_setup.save_test(run_test, mse_dict, save_img_seq= save_img_seq, unique_csv=unique_csv)

    elif run_test==2:
        query1 = pred_data[:,15:,...]
        # query2 = pred_data[:,8,...].unsqueeze(1)
        
        # queries = cat(query1, query2)
        # query3 = pred_data[:,13,...].unsqueeze(1)
        
        # queries = cat(queries, query3)

        
        pred_data = query1

        print('pred data', pred_data.shape)

        if model_config == 'GQN_OZ' or gqn_oz_test_type == True:
            pass
        elif model_config == 'GQN':
            
            pred_seq = TensorDict({})
            for k in pred_data.keys():
                pred_seq[k+'_query'] = pred_data[k]

            models[0].reset()
            standard_normal = MultivariateNormal(torch.zeros(n_sample,32),torch.ones(n_sample,32)).unsqueeze(1)
            future_step = models[0].forward(pred_seq, standard_normal, reconstruct=True)
            future_step['image_reconstructed'] = future_step['image_predicted']
           
            
            for step in range(future_step['image_predicted'].shape[1]):
                #for sample in range(images_comparison['image_predicted'].shape[0]):
                mse_by_step = torch.nn.functional.mse_loss(future_step['image_predicted'][:,step,...], pred_seq['image_query'][:,step,...], reduction='mean').cpu().detach().numpy().tolist()

                pred_img = vis_images(torch.mean(future_step['image_predicted'][:,step,...], dim=0).unsqueeze(0),show=False, fmt="torch").cpu().detach().numpy()
                query_img = vis_images(torch.mean(pred_seq['image_query'][:,step,...] , dim=0).unsqueeze(0),show=False, fmt="torch").cpu().detach().numpy()
                
                print('zese5utf',pred_img.shape)
                pred_img = np.transpose(pred_img, (1, 2, 0))
                query_img = np.transpose(query_img, (1, 2, 0))
                squared_error = (query_img- pred_img)**2
                

                squared_error = squared_error.sum(axis= 2 ) 
                plt.clf()
                plt.imshow( squared_error , vmin=0, vmax=3)
                plt.title(str(step) + '_mse:'+ str(mse_by_step))
                plt.colorbar()
                plt.savefig('test_results/'+ str(step))
            #mean mse of all images reconstructed vs groundtruth
            mse2 = test_setup.compute_mse_error(future_step, queries= pred_data,  show_img = False)
            
            
            #print('check', mse2)
            ##mean std of the posterior at each step##
            mean_std_place = test_setup.compute_std_dist(standard_normal)['mean_std_dist'][0]

            m_output_seq, model, result_dict = run_GQNmodel_step_by_step(models, model_set_data, pred_data, test_setup)
            
            result_dict['place_std_per_step'].insert(0,mean_std_place)
            result_dict['mse_step'].insert(0,mse2['mse_image']*10)

            print('result dict', result_dict['mse_step'])
        elif model_config == 'Conv' :
            pass
                
        if save_img_seq == True:
            steps=[]
            i=0
            n=1
            print(model_set_data.shape)
            while i < model_set_data.shape[1]:
                steps.append(i)
                i+=n
                #n+=1
            print(steps)
                
            test_setup.save_model_seq_steps(predicted_image_seq = m_output_seq['image_reconstructed'], img_query =pred_data,  steps=steps, save_model_steps = True) 
            test_setup.save_model_mean_steps_pred(predicted_image_seq = m_output_seq['image_reconstructed'], img_query =pred_data, steps=steps, mse_steps = result_dict['mse_step'])   

        test_setup.save_test(run_test, result_dict, save_img_seq= save_img_seq, unique_csv=unique_csv)
    
    elif run_test ==3 :
        #Poses options in gridworld (could go further in aisle, but well)
        x = list(range(-5, 6))
        y =  list(range(-5, 6))
        theta = list(range(0, 4))
        pose_options = list(product(*[x,y,theta]))
        pose_options = list(map(list, pose_options)) #all pose options as a list of list
        print('len(pose_options', len(pose_options))
        result_dict = {}
        reduced_seq = model_set_data[:,:4]
        reduced_seq_2 = model_set_data[:,4:5]
        pose_options.insert(0,reduced_seq_2['pose'][0].cpu().detach().numpy().tolist()[0]) #first value IS the correct one


        #print(model_set_data['pose'][0])
        if model_config == 'GQN_OZ' or gqn_oz_test_type == True:
            pass
        elif model_config == 'GQN':
            for i in range(0,len(pose_options)):
                pose_predicted = {'prev_pose': reduced_seq['pose'][0].cpu().detach().numpy().tolist(), 'real_pred_pose': pose_options[0] , 'pred_pose':pose_options[i]}
                result_dict.update(pose_predicted)
                pose = torch.Tensor(pose_options[i]).unsqueeze(0).repeat(reduced_seq_2['image'].shape[0],1,1)
                #print(reduced_seq_2.pose.shape, pose.shape, pose[0])
                reduced_seq_2['pose'] = pose
                #print(reduced_seq['pose'].shape)
                m_output, models = run_model(models,reduced_seq)
                pred_output = run_GQN_prediction_model(m_output, models, reduced_seq_2)
                #Compute error in the benchmark and save data in a csv file
                #print(pred_output['image_reconstructed'].shape, m_output.place.shape)
                mse_dict=test_setup.compute_mse_error_one_step(pred_output, reduced_seq_2, key=['image'], show_img = show_img)
                result_dict.update(mse_dict)
                            
                m_output_step2, models = run_model(models,reduced_seq_2, **{'reset': False})
                if 'place' in pred_output:
                    prev_post = m_output.place
                    post = m_output_step2.place
                else:
                    prev_post = m_output.posterior
                    post = m_output_step2.posterior

                #Compute surprise between 2 steps
                #print(post.shape, type(post), prev_post.shape)
                
                kl_dict = test_setup.compute_KL(state_dist_seq= post, pref_dist_seq=prev_post)
                kl_dict['kl_by_step'] = kl_dict['kl_by_step'][0]
                result_dict.update(kl_dict)
                result_dict.update(idx = i)
                test_setup.save_test(run_test, result_dict, save_img_seq= save_img_seq, unique_csv=unique_csv)

            
        elif model_config == 'Conv' :
            pass

    elif run_test == 6:
        standard_normal = MultivariateNormal(torch.zeros(n_sample,32),torch.ones(n_sample,32)).unsqueeze(1)

        poses_query = TensorDict({'pose_query': torch.tensor([[0.,0.,0.],[0.,0.,1.], [0.,0.,2.], [0.,0.,3.]]).unsqueeze(0).repeat(n_sample,1,1)})
        if model_config == 'GQN':
            step = models[0].forward(poses_query, place=standard_normal, reconstruct=True)
        
        vis_images(step['image_predicted'], show=True, fmt="torch", title="imagined views for p [4,4,0],[4,4,1], [4,4,2], [4,4,3] ")



        
def run_model(model, sequence, **kwargs):
    '''
    should work for all models as long as they authorise kwargs as fct call input and pass it to the forward fct 
    '''
    
    
    dict = {'train': False, 'reconstruct':False}
    output = model[0](sequence, **dict, **kwargs)
    
    return output, model


def run_GQNmodel_step_by_step(model,sequence, prediction_sequence, test_setup):
    '''
    should work for all models as long as they authorise kwargs as fct call input and pass it to the forward fct 
    '''
    res = None
    results_dict = {}
    #the summary created a prior in model
    model[0].reset()

    pred_seq = TensorDict({})
    for k in prediction_sequence.keys():
        pred_seq[k+'_query'] = prediction_sequence[k]
    #print(pred_seq['pose_query'].shape, sequence['pose'].shape)
    for step in range(sequence.shape[1]):
        
        data= sequence[:,step,...].unsqueeze(1)
        data.update(pred_seq) #Add the pred sequence to data to get the prediction with post at each step
        output = model[0].forward(data, reconstruct=True)
        output['image_reconstructed'] = output['image_predicted']
        if 'pose_predicted':
            output['pose_reconstructed'] = output['pose_predicted']
        #print('output rec', output['image_reconstructed'].shape, output['image'].shape)      
        
        if 'place' in output:
            place = output['place'] 
        elif 'posterior' in output:
            place = output['posterior']

        #mean mse of all images reconstructed vs groundtruth
        mse = test_setup.compute_mse_error(output, queries= prediction_sequence,  show_img = False)['mse_image']*10
        

        #mean std of the posterior at each step
        mean_std_place = test_setup.compute_std_dist(place)['mean_std_dist'][0]
               
        o = TensorDict(output).unsqueeze(1)       
        if res is None:
            res = o
            results_dict['place_std_per_step'] = [mean_std_place]
            results_dict['mse_step'] = [mse]
            
        else:
            res = cat(res, o) #OUTPUT : [batch, temporal_seq, number_of_elements_in_this_t, ... ]  #number_of_elements_in_this_t--> Linked to image_reconstructed
            results_dict['place_std_per_step'].append(mean_std_place)
            results_dict['mse_step'].append(mse)
        #print(res['state'].shape, res['image'].shape, res['image_reconstructed'].shape)
           
    return res, model, results_dict

        
def run_GQN_OZ_prediction_model(m_output, model, sequence):
    output = TensorDict({'place': m_output['place']}) #accumulated posterior
    res = None
    seq = TensorDict({})
    for k in sequence.keys():
        seq[k+'_query'] = sequence[k]
    #to have a prediction based on only present and past info, this shouldn't be necessary considering all archi, but just to be safe
    for timestep in range(seq.shape[1]):
        step = seq[:, timestep]
        step.unsqueeze(1)
        output = model[0].forward(step, place=output['place'], reconstruct=True)
        
        if len(model) == 2: #Ozan model is not in the forward function of model[0]
            output.update(model[1].likelihood(output['state_predicted'].sample(), 'image'))
    
        o = TensorDict(output.copy()).unsqueeze(1)
        if res is None:
            res = o
        else:
            res = cat(res, o)
        
    return res
        
def run_GQN_prediction_model(m_output, model, sequence):

    if 'place' not in m_output:
        old_GQN_archi = True
    else:
        old_GQN_archi= False

    if old_GQN_archi:
        output = TensorDict({'posterior': m_output['posterior']}) #accumulated posterior
    else:
        output = TensorDict({'place': m_output['place']}) #accumulated posterior
    
    seq = TensorDict({})
    for k in sequence.keys():
        seq[k+'_query'] = sequence[k]
        
    if old_GQN_archi:
        output = model[0].forward(seq, latent_place=output['posterior'], reconstruct=True)
    else:
        output = model[0].forward(seq, place=output['place'], reconstruct=True)
           
    output['image_reconstructed'] = output['image_predicted']
    if 'pose_predicted' in output:
        output['pose_reconstructed'] = torch.round(output['pose_predicted'])
    return output



def run_OZ_prediction_model(model, sequence):
    res = None
    
    for timestep in range(sequence.shape[1]):
        step = sequence[:, timestep,...]
        
        output = model[0].forward(action= step['action'], reconstruct=True)
        
        o = TensorDict(output.copy()).unsqueeze(1)
        if res is None:
            res = o
        else:
            res = cat(res, o)

    return output
        