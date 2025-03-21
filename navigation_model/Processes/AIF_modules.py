import torch
import numpy as np
from dommel_library.modules.dommel_modules import tensor_dict


# ------------ AIF FUNCTIONS ------------------# 
def calculate_FE_EFE(model: object, place:torch.Tensor, model_step_output:dict, preferred_state:torch.Tensor) -> tuple[float, float, float, np.ndarray]:
    ''' check FE at each step and predict one step EFE
    with current :
    model (Class)
    model one model_step_output output (TensorDict
    )
    preferred_state we want to favorise (MultivariateNormal)
    '''
    KL,ambiguity= KL_and_ambiguity(model_step_output, preferred_state = preferred_state)
    mse, state_prob_in_place, image_predicted_data = free_energy(model, place, model_step_output, preferred_state = preferred_state)

    return KL[0], mse, np.mean(state_prob_in_place), image_predicted_data

def free_energy( model:object, place:torch.Tensor, sequence:dict, preferred_state:torch.Tensor = None)-> tuple[float,list,np.ndarray]:
    mse,image_predicted_data = mse_observation(model, place,sequence) # only works with real observation given with pose
    state_prob_in_place = logprob_observation(model, sequence,preferred_state)

    return float(mse.detach().cpu().numpy()), state_prob_in_place.tolist(), image_predicted_data

def estimate_a_surprise(state:torch.Tensor, preferred_state:torch.Tensor, predicted_obs:torch.Tensor, preferred_ob:torch.Tensor):
    """
    Get KL:
    given a distribution : state
    a distribution of comparison : preferred_state
    Get ambiguity between predicted ob:
    images of format [sample, ...]: predicted_obs
    Get discrepency between those predicted obs and an image of reference: preferred_ob [sample,...]
    """
    ambiguity = observation_ambiguity(predicted_obs).detach().cpu().numpy().tolist()[0]
    kl = round(np.mean(calculate_KL(state, preferred_dists=preferred_state).detach().cpu().numpy()), 4)
    pred_expected_error = mse_elements(predicted_obs, preferred_ob).detach().cpu().numpy().tolist()
    return kl, ambiguity, pred_expected_error

def KL_and_ambiguity(imagined_sequence,preferred_state=None, ambiguity_beta=1):
    """ expected entry shape [batch,lookahead,dims], outputs shape [lookahead,1] """
    
    if 'place' in imagined_sequence:
        posterior = imagined_sequence.place
    elif 'posterior' in imagined_sequence:
        posterior = imagined_sequence.posterior
    elif 'prior' in imagined_sequence:
        posterior = imagined_sequence.prior

    
    #just for test
    if 'image_predicted' in imagined_sequence:
        keys = ["image_predicted"]
    if 'image_reconstructed' in imagined_sequence:
        keys = ["image_reconstructed"]
    
    elif 'image' in imagined_sequence:
        keys = ["image"]
    elif 'state' in imagined_sequence:
        keys = ['state']
        
    else:
        raise KeyError ('NO IDEA WHICH KEY TO EXTRACT FOR logprob_reconstruction')

    logprob_image = logprob_reconstruction(imagined_sequence, keys).detach().cpu().numpy().tolist()
    if not isinstance(logprob_image, list):
        logprob_image = [logprob_image]   
                                
    epistemic_term = calculate_KL(posterior, preferred_state).detach().cpu().numpy() 
    if isinstance(epistemic_term,float):
        epistemic_term = [epistemic_term]   
    epistemic_term = epistemic_term* ambiguity_beta
   
    return  epistemic_term.tolist(),logprob_image

def compute_std_dist(dist)->float:
    ''' 
    extract the std from either a distribution as an array or tensor.
    calculate the mean std of this distribution over dimensions and return it
    '''
    std_dist = dist[...,(dist.shape[-1] // 2):]
    mean_std_dist = torch.mean(1 / 2 * torch.log(2 * np.pi * np.e * std_dist ** 2), dim=[0,-1]).cpu().detach().numpy().tolist()[0]
    return mean_std_dist

def calculate_KL(state_dist, preferred_dists=None):
    KL = torch.tensor([0.0]) 
    #print('before state mean shape + preferred dist shape', state_dist.shape, preferred_dists.shape)
    if preferred_dists is not None:
        mean_state_dist = torch.mean(state_dist, dim=0)
        if len(mean_state_dist.shape) == 1:
            mean_state_dist = mean_state_dist.unsqueeze(0)
        if len(preferred_dists.shape) == 1:
            preferred_dists = preferred_dists.unsqueeze(0).to(state_dist.device)
        elif len(preferred_dists.shape) > 2:
            preferred_dists =  torch.mean(preferred_dists, dim=0).to(state_dist.device)
        #print('state mean shape + preferred dist shape', mean_state_dist.shape, preferred_dists.shape)
        KL = torch.distributions.kl_divergence(mean_state_dist, preferred_dists)
        KL = KL / preferred_dists.shape[-1]
    #print('KL CHECK',KL, type(KL))

    return KL

def logprob_reconstruction(sequence, key=[]): 
    ''' check  prior reconstructed ob ambiguity '''
    H = 0   
    for k in key: #not adapted for more than 1 key
        ob = sequence[k]
    #prior = torch.mean(prior, dim=0)
        if len(ob.shape) == 4 : #IMAGE only [sample,lookahead,3,64,64], if <5, no lookahead
            ob = ob.unsqueeze(1)
        sigmas = torch.std(ob, dim=0)
        Hs = 1 / 2 * torch.log(2 * np.pi * np.e * sigmas ** 2)
        #print('Hs', Hs.shape)

        #to consider images shape
        if len(Hs.shape) > 2:
            H = torch.mean(Hs, dim=list(range(1,len(Hs.shape))))
        else:
            H = torch.mean(Hs, dim=-1)
    return H

def observation_ambiguity(ob:torch.Tensor)->torch.Tensor:
    ''' check if the different predictions match each other or not '''
    H = 0   
    if len(ob.shape) == 4 : #IMAGE only [sample,lookahead,3,64,64], if <5, no lookahead
        ob = ob.unsqueeze(1)
    sigmas = torch.std(ob, dim=0)
    Hs = 1 / 2 * torch.log(2 * np.pi * np.e * sigmas ** 2)

    #to consider images shape
    if len(Hs.shape) > 2:
        H = torch.mean(Hs, dim=list(range(1,len(Hs.shape))))
    else:
        H = torch.mean(Hs, dim=-1)
    return H

def logprob_observation(model:object, sequence:dict, pref_post:torch.Tensor)->torch.Tensor: 
    ''' check  log prob of seq stste in preferred post distrib'''
    logprob_preferences = 0
    
    with torch.no_grad():
        model.reset()
        step = model.forward(sequence, place = None, reconstruct=False) 

    logprob_preferences = pref_post.log_prob(step['state']) / np.prod(pref_post.shape)
    
    return logprob_preferences

def mse_observation(model:object, place:torch.Tensor, sequence:dict)-> tuple[float,np.ndarray]: 
    ''' check  expected ob with current pose and post vs real ob '''
    
    tmp_seq = tensor_dict({'pose_query': sequence['pose'] })
    with torch.no_grad():
               
        step = model.forward(tmp_seq, place, reconstruct=True) 
    model_error = mse_elements(step['image_predicted'], sequence['image'])
    return model_error, step['image_predicted']

def mse_elements(prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor: 
    ''' check mse between 2 elements '''
    model_error = torch.nn.functional.mse_loss(prediction, target, reduction='mean') *10 #/ np.prod(sequence['image'].shape) *10
    return model_error

