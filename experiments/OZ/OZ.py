import os
import torch
from torchvision.transforms import Compose
import glob
import copy

from dommel_library.datasets.dataset_factory import dataset_factory
from dommel_library.datasets import MemoryPool
from dommel_library.datasets.transforms import (
    Resize, ChannelFirst, RescaleShift, ToFloat,
    Squeeze, Subsample, Unsqueeze, Pad, Crop)
from dommel_library.datastructs import Dict, TensorDict, cat as datastructs_cat
from dommel_library.nn import module_factory
from dommel_library.nn.summary import summary
from dommel_library.train import (
    Trainer,
    loss_factory,
    optimizer_factory,
)
from dommel_library.modules.visualize import visualize_sequence
from .models import ConvModel, BaseModel
from .train import ModelTrainer

# from test_benchmark import Benchmark

def get_model_parameters(log_dir, epoch=None):
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


class ConsecutiveDuplicateCheck():
    """ Check 2 consecutives poses to check if there is a collision """

    def __init__(self, keys, new_data_names):
        self.keys = keys
        self.new_data_names = new_data_names

    def __call__(self, sequence):
        new_data_names = self.new_data_names.copy()
        for key in self.keys:
            if key in sequence.keys():
                #print('sequence[key].shape for key pose', sequence[key].shape)
                duplicate_bool_list = [0.]
                
                for x in range(1,sequence[key].shape[0]):
                    #print(sequence[key][x-1],sequence[key][x], (sequence[key][x-1] == sequence[key][x]).all())
                    if (sequence[key][x-1] == sequence[key][x]).all():
                        duplicate = 1.
                    else:
                        duplicate = 0.
                    duplicate_bool_list.append(duplicate) 
                
                duplicate_tensor = torch.tensor(duplicate_bool_list).unsqueeze(1)
                #print('duplicate_tensor and sequence pose shape', duplicate_tensor.shape, sequence[key].shape)
                sequence[new_data_names[0]] = duplicate_tensor
                new_data_names.pop(0)
        return sequence

class DelKey():
    """ del input key"""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sequence):
        for key in self.keys:
            if key in sequence.keys():
                del sequence[key]
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
    
    if 'image' in config.dataset['keys']:
            size_cam = config.model.observations.image
            if isinstance(size_cam,dict):
                size_cam = size_cam['input_shape']
    else:
        size_cam = [3,64,64]
    
    #print('size_cam', size_cam)

    train_destination = config.dataset.get('train_destination',None)
    val_destination = config.dataset.get('val_destination',None)
        
    transforms = [ToFloat(),
                  #Unsqueeze(0, keys=['vel_ob']),
                  RescaleShift(1.0 / 255, 0, keys=["image"]),
                  ChannelFirst(keys=["image"]),
                  Resize(keys=["image"], size=size_cam),
                  ConsecutiveDuplicateCheck(['pose'], ['collision']),
                  DelKey(keys=['pose'])
                  ]
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

    model_config = config.model
    
    model_type = model_config.get("type", None)
    if model_type == "Conv":
        model = ConvModel(device = config.device, **model_config)
    elif model_type == "Base":
        model = BaseModel(**model_config)
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
            model.load(get_model_parameters(path, epoch = epoch), map_location='cpu')
        elif "params_dir" in config.keys():
            model.load(get_model_parameters(config.params_dir, epoch = epoch), map_location='cpu')
        else:
            print('config log dir',config.log_dir)
            print('epoch,', epoch)
            model.load(get_model_parameters(config.log_dir, epoch= epoch), map_location='cpu')
    
    model.to(config.device)
    print(model)
    
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
    model_config = config.model.get("type", None)
    
    model = create_model(Dict(copy.deepcopy(config)), False)

    models =[model]
    
    if 'scenario' in config.dataset and scenario== None:
        config.dataset.val_set_location = config.dataset.scenario
    elif scenario != None:
        config.dataset.val_set_location = scenario


    #-- Collect Dataset --
    config.dataset.sequence_length = -1
    config.dataset.sequence_stride = -1
    _, eval_set = create_datasets(config, False)
    print('config.log_dir',config.log_dir)


    test_setup = Benchmark(model_id = config.log_dir.split('/')[-1], data_file= config.dataset.val_set_location)
    
    if run_test ==1:
        model_set_data, pred_data = test_setup.parse_data(eval_set, n_sample=n_sample, learn_steps= learn_steps, lookahead=lookahead, lookahead_ratio=lookahead_ratio)

        #Run model from call
        m_output, models = run_model(models, model_set_data)

        
        if model_config == 'Conv' or model_config == 'RSSM':
            pred_output = run_OZ_prediction_model(models, pred_data)
        else :
            print('model %s not recognised', model_config)


        #Compute error in the benchmark and save data in a csv file
        mse_dict= test_setup.compute_mse_error(pred_output, key=['image', 'collision'], show_img = show_img)
        test_setup.save_test(run_test, mse_dict, save_img_seq= save_img_seq, unique_csv=unique_csv)

def run_model(model, sequence):
    '''
    should work for all models as long as they authorise kwargs as fct call input and pass it to the forward fct 
    '''
    model[0].reset()
    dict = {'train': False, 'reconstruct':False}
    #print(' sequence[state', sequence['state'].shape)
    # for step in range(sequence.shape[1]):
    #     images = model[0].likelihood(sequence['state'][:,step,...], 'image')
    #     images.update(image = sequence['image'][:,step,...])
    #     print('image shape', images['image'].shape)
    #     visualize_sequence(images.unsqueeze(1),
    #                         ["image", 'image_reconstructed'], 
    #                         show=True,
    #                         title="query and predicted sequence")

    output = model[0](sequence)
    
    return output, model


def run_OZ_prediction_model(model, sequence):
    res = None
    
    for timestep in range(sequence.shape[1]):
        step = sequence[:, timestep,...]
        
        #output = model[0].likelihood(step['state'], 'image')
        output = model[0].forward(action= step['action'], reconstruct=True)
        if not 'image_reconstructed' in output:
            output['image_reconstructed'] = output['image']

        o = TensorDict(output.copy()).unsqueeze(1)
        if res is None:
            res = o
        else:
            res = datastructs_cat(res, o)
        

    return res

#------ old ------

def old_evaluate(config):
    print("Loading data ...")
    if 'scenario' in config.dataset:
        config.dataset.val_set_location = config.dataset.scenario
    _, eval_set = create_datasets(config, False)
    model = create_model(config, False)

    sequence = eval_set.sample()
    evaluation_plots(model,sequence)


def evaluation_plots(model,sequence):
    ''' Display as image the observation seq, reconstruction seq and prior seq'''
    # visualize sequence
    visualize_sequence(sequence,
                       ["image"],
                       max_length=10,
                       vis_mapping={"image": ["image"]},
                       show=True,
                       title="Example sequence")


    # visualize posterior reconstruction
    sequence_posterior = model_rollout(model, sequence).squeeze(0)
    visualize_sequence(sequence_posterior,
                        ["image"],
                        max_length=10,
                        vis_mapping={"image": ["image"]},
                        show=True,
                        title="Posterior reconstructions")
    model.reset()
    model_copy = copy.deepcopy(model)
    #Visualise prior
    sequence_prior = imagine_test(model_copy, sequence.action)
    visualize_sequence(sequence_prior,
                    ["image"],
                    max_length=10,
                    vis_mapping={"image": ["image"]},
                    show=True,
                    title="Prior reconstructions")

def model_rollout(model, sequence,reset=True):
    ''' Calculate a sequence of prior and posterior distributions
        and likelihoods for a given sequence
        `model`               - the model with a prior and likelihood function
        `sequence`            - the sequence to process
    '''
    result = []
    sequence = sequence.to(model.device)
    if reset:
        model.reset()
    for j in range(sequence.action.shape[1]):
        action = sequence.action[:, j, :]
        observations = {k: sequence[k][:, j, :] for k in model.observations()}
        #vis_image(observations['image'].squeeze(0), show=True)
        
        step = model.forward(action, observations)
        #print(step.image.squeeze(0).shape)
        
        result.append(TensorDict(step).unsqueeze(1))

    return datastructs_cat(*result)


def imagine_test(model_copy, action_seq):
    imagined_sequence = []
    
    for step in range(action_seq.shape[1]):
        action = action_seq[:,step, :]
        step = model_copy.forward(action = action)
        step_storage = TensorDict({})
        step_storage.update(step)
        step_storage.action = action

        imagined_sequence.append(step_storage.unsqueeze(1))
    return datastructs_cat(*imagined_sequence)