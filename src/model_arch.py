from torch import nn
import argparse




def get_model_arch(arch_name, layer_params):

    # Init config
        if arch_name == 'nn':
        encoder_config = OrderedDict({'lin1_encoder': nn.Linear(layer_params['encoder_in'], layer_params['encoder_out']),
                                      'norm1_encoder': nn.BatchNorm1d(layer_params['encoder_out']), 'relu1_encoder': nn.ReLU()})

        decoder_config = OrderedDict({'lin1_decoder': nn.Linear(layer_params['decoder_in'], layer_params['decoder_out']),
                                      'norm1_decoder': nn.BatchNorm1d(layer_params['decoder_out']), 'relu2_decoder': nn.ReLU()})

        classifier_config = OrderedDict({'lin1_classifier': nn.Linear(layer_params['classifier_in'], layer_params['classifier_out']),
                                         'sig_classifier': nn.Sigmoid()})
    elif arch_name == 'conv':
        encoder_config = OrderedDict({'unflatten': nn.Unflatten(1, (1, layer_params['input_dim'])),
                                      'lin1_encoder': nn.Conv1d(layer_params['encoder_in_channels'], layer_params['encoder_out_channels'], kernel_size=layer_params['encoder_kernel'], stride=layer_params['stride']),
                                        'flatten': nn.Flatten(), 'norm1_encoder': nn.BatchNorm1d(layer_params['decoder_in']), 'relu_encoder': nn.ReLU()})

        decoder_config = OrderedDict({'lin1_decoder': nn.Linear(layer_params['decoder_in'], layer_params['decoder_out']),
                                      'norm1_decoder': nn.BatchNorm1d(layer_params['decoder_out']), 'relu_decoder': nn.ReLU()})

        classifier_config = OrderedDict({'linear1_classifier': nn.Linear(layer_params['classifier_in'], layer_params['classifier_out']), 'sig_classifier': nn.Sigmoid()})

    elif arch_name == 'nn-sig':
        encoder_config = OrderedDict({'lin1_encoder': nn.Linear(layer_params['encoder_in'], layer_params['encoder_out']),
                                      'norm1_encoder': nn.BatchNorm1d(layer_params['encoder_out']), 'sig_encoder': nn.Sigmoid()})

        decoder_config = OrderedDict({'lin1_decoder': nn.Linear(layer_params['decoder_in'], layer_params['decoder_out']),
                                      'norm1_decoder': nn.BatchNorm1d(layer_params['decoder_out']), 'sig_decoder': nn.Sigmoid()})

        classifier_config = OrderedDict({'lin1_classifier': nn.Linear(layer_params['classifier_in'], layer_params['classifier_out']),
                                         'sig_classifier': nn.Sigmoid()})

    else:
        raise Exception('Incorrect model type: use \'conv\' or \'nn\' or \'nn-sig\'')


    return encoder_config, decoder_config, classifier_config




def main(**kwargs):
    
    # Get configurations
    arch_name = kwargs['arch_name']
    layer_params = kwargs['layer_params']
    encoder_config, decoder_config, classifier_config = get_model_arch(arch_name, layer_params)
    
    # Print config
    print()

if __name__ == '__main__':
    # Parse args

    # Call main
    main()
