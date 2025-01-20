from dommel_library.datastructs import TensorDict, cat 




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

    return cat(*result)

def imagine_rollout(model,
                    agent,
                    max_sequence_length=100):
    '''
    Imagine a single experience sequence of length `max_sequence_length`
    by doing a role-out with the agent and a model's prior and likelihood,
    the agent and model should be compatible
    Arguments:
    `model`               - the model with a prior and likelihood function
    `agent`               - an agent compatible with the environment
    `max_sequence_length` - the max length of a sequence, default 100
    '''
    result = []
    agent.reset()

    step = TensorDict({})
    step.state = model.get_state()
    action, _ = agent.act(step)

    for _ in range(max_sequence_length):
        t = model.forward(action.to(model.device))
        step = TensorDict({})
        step.action = action
        
        step.update(t)

        action, _ = agent.act(step)
        result.append(step.unsqueeze(1))

    return cat(*result)






class ReplayAgent():

    def __init__(self, actions, start_index=0):
        super().__init__()
        self.actions = actions
        self.start_index = start_index
        self.i = self.start_index

    def action_size(self):
        return self.actions.shape[-1]

    def act(self, observations):
        action = self.actions[:, self.i, ...]
        if self.i < self.actions.shape[1] - 1:
            self.i += 1
        return action, TensorDict({})

    def reset(self):
        self.i = self.start_index
