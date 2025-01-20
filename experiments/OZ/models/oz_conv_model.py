import torch
from dommel_library.nn import (
    ConvPipeline,
    UpConvPipeline,
    Cat,
    VariationalMLP,
    VariationalGRU,
    MLP,
    VariationalLSTM,
    Activation,
    get_activation,
)
from dommel_library.nn.composable_module import ComposableModule
from dommel_library.datastructs import Dict

from .oz_base_model import BaseModel


class ConvModel(BaseModel):

    def __init__(
        self,
        num_states,
        num_actions,
        observations,
        channels = [],
        hidden_layers = None,
        lstm_cells=None,
        lstm_posterior=False,
        gru_cells=None,
        gru_posterior=False,
        conv_block="Conv",
        decoder_film=False,
        decoder_act=None,
        device="cpu",
        **kwargs
    ):
        # observation_type=[]
        # observation_input=[]
        # for key, value in observations.items():
        #     observation_type.append(key)
        #     observation_input.append(value)


        blocks = ["Conv", conv_block, ...]

        prior_cat = Dict({"module": Cat(dim=1),
                          "input": ["action", "state"],
                          })

        # prior_lin = Dict({"module":
        #                     MLP(
        #                         num_states,
        #                         num_features,
        #                         hidden_layer,
        #                         activation=dict_input.get(
        #                             "activation", "Activation")),
        #                     "input": "state",
        #                     })
        if lstm_cells is not None:
            prior_nn = Dict({"module":
                             VariationalLSTM(num_states + num_actions,
                                             num_states,
                                             lstm_cells),
                             "input": ["...", "hidden"],
                             "output": ["prior", "hidden"],
                             })
        elif gru_cells is not None:
            prior_nn = Dict({"module":
                             VariationalGRU(num_states + num_actions,
                                            num_states,
                                            gru_cells),
                             "input": ["...", "hidden"],
                             "output": ["prior", "hidden"],
                             })
        else:
            prior_nn = Dict({"module":
                             VariationalMLP(
                                 num_states + num_actions,
                                 num_states,
                                 [hidden_layers[0], hidden_layers[0]],
                                 activation=kwargs.get(
                                     "activation", "Activation")),
                             "output": "prior",
                             })                   
        
        prior = ComposableModule([prior_cat,
                                  prior_nn])

       
        posterior_conv=[]
        posterior_input= ["action", "state"]
        output_types = []
        posterior_num_features=0

        likelihoods = torch.nn.ModuleDict()
        #if several sensors observations
        observation_dict = {}
        for observation_type, value in observations.items():
            
            #if we have different models for each observation
            if isinstance(value,(dict, Dict)):
                #TODO: THIS PART WAS NOT TESTED IN REAL SCENARIOS
                #observation_type = next(iter(value['observations']))
                type = value['type']
                input_shape = value['input_shape'] #value['observations'][next(iter(value['observations']))]
                del value['input_shape']
                if 'channels' in value:
                    channel = value['channels']
                    del value['channels']
                else:
                    channel = channels
                
                if 'hidden_layers' in value:
                    hidden_layer = value['hidden_layers']
                    del value['hidden_layers']
                else:
                    hidden_layer = hidden_layers

                dict_input = value
                
                
            else:
                type = 'Conv'
                input_shape = value
                dict_input = kwargs
                channel = channels
                hidden_layer = hidden_layers
            output_type = "features_"+str(observation_type) 
            observation_dict[observation_type] = input_shape
            if type == 'Conv':
                posterior_conv.append(Dict({"module":
                                    ConvPipeline(input_shape,
                                                    channel,
                                                    block=blocks,
                                                    **dict_input
                                                    ),
                                    "input": observation_type,
                                    "output": output_type,
                                    }))
                num_features = posterior_conv[-1].module.output_length
                posterior_num_features+= num_features
                output_types.append(output_type)
                posterior_input.append(output_type)   
               
                # stride becomes interpolate in decoder
                if "stride" in dict_input.keys():
                    stride = dict_input["stride"]
                    if isinstance(stride, list):
                        interpolate = stride[::-1]
                    else:
                        interpolate = stride
                    dict_input["interpolate"] = interpolate
                    del dict_input["stride"]

                if decoder_film:
                    likelihood_conv = Dict({"module":
                                            UpConvPipeline(
                                                input_shape, #input_shape
                                                channel[::-1],
                                                block=blocks[::-1],
                                                condition_size=num_states,
                                                **dict_input
                                            ),
                                            "input": ["...", "state"],
                                            "output": observation_type+"_reconstructed",
                                            })
                else:
                    likelihood_conv = Dict({"module":
                                            UpConvPipeline(
                                                input_shape,
                                                channel[::-1],
                                                block=blocks[::-1],
                                                **dict_input
                                            ),
                                            "output": observation_type+"_reconstructed",
                                            })
                likelihood_connect = Dict({"module": Activation()})
            
            elif type == 'Bool':
                num_features = input_shape
                likelihood_connect = Dict({"module": #TODO: I WANT SIGMOID WITH TH 0.5 -doesn't exist TT-
                                    get_activation(activation= dict_input.get('mlp_activation', Activation()))}) 
                likelihood_conv = Dict({"module": #TODO: I want NOTHING, doing twice same activation == to doing nothing
                                    get_activation(activation= 'Identity'), 
                                    "output": observation_type+"_reconstructed"}) 
            

            likelihood_mlp = Dict({"module":
                            MLP(
                                num_states,
                                num_features,
                                hidden_layer,
                                activation=dict_input.get(
                                    "activation", "Activation")),
                            "input": "state",
                            })
            

            likelihood_layers = [likelihood_mlp,
                                likelihood_connect,
                                likelihood_conv
                                ]
            if decoder_act:
                likelihood_layers.append(
                    Dict({"module": get_activation(decoder_act)}))
                        
            likelihoods[observation_type]= ComposableModule(likelihood_layers)

        #posterior_cat = Dict({"module": Cat(dim=1),
        #                    "input": posterior_input
        #                    })
        
        # posterior_mlp = Dict({"module":
        #                     VariationalMLP(
        #                         posterior_num_features + num_states + num_actions,
        #                         num_states,
        #                         hidden_layers,
        #                         activation=kwargs.get(
        #                             "activation", "Activation")),
        #                     "output": "posterior",
        #                     })
       
        if not (lstm_posterior or gru_posterior):
        
            posterior_cat = Dict({"module": Cat(dim=1),
                                    "input": posterior_input,
                                    })
            posterior_mlp = Dict({"module":
                                    VariationalMLP(
                                        posterior_num_features + num_states + num_actions,
                                        num_states,
                                        hidden_layers,
                                        activation=kwargs.get(
                                            "activation", "Activation")),
                                    "output": "posterior",
                                    })
        else:
            if lstm_posterior:
                hidden_key = "hidden" + str(len(lstm_cells) - 1)
                num_hidden = lstm_cells[-1]

            elif gru_posterior:
                hidden_key = "gru" + str(len(gru_cells) - 1)
                num_hidden = gru_cells[-1]
            
            output_types.insert(0,hidden_key)
            posterior_cat = Dict({"module": Cat(dim=1),
                                "input": output_types,
                                })
     
            posterior_mlp = Dict({"module":
                                VariationalMLP(
                                    posterior_num_features + num_hidden,
                                    num_states,
                                    hidden_layers,
                                    activation=kwargs.get(
                                        "activation", "Activation")),
                                "output": "posterior",
                                })

        
        modules_list = posterior_conv.copy()           
        modules_list.append(posterior_cat)
        modules_list.append(posterior_mlp)
        posterior=ComposableModule(modules_list)
        
           
        BaseModel.__init__(self,
                           num_states,
                           num_actions,
                           observation_dict,
                           prior,
                           posterior,
                           likelihoods,
                           device="cpu",
                           **kwargs)
