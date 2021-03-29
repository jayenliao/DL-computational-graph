# Experiment of different activation functions of the hidden layer

import pickle
import numpy as np
import matplotlib.pyplot as plt
from args import init_arguments
from source.utils import smooth_curve
from main import main

if __name__ == '__main__':
    model = 'TwoLayerNet'
    parser = init_arguments()
    parser.add_argument('--hidden_act_fun_list', nargs='+', type=str, default=['ReLU', 'Sigmoid', 'tanh'])
    args = parser.parse_args()
    args.models = [model]
    feature_type = args.feature_types[0]
    
    d_train_loss = {}
    for fun in args.hidden_act_fun_list:
        print('----> Activation function of the hidden layer =', fun, '\n')
        args.hidden_layer_act = fun
        trainer = main(args, feature_type=feature_type, return_trainer=True)[model]
        d_train_loss[fun] = trainer.train_loss

    fn = trainer.fn.replace('Accuracy', 'dict_loss_hidden_act').replace('.txt', '.pkl')
    with open(fn, 'wb') as f:
        pickle.dump(d_train_loss, f, pickle.HIGHEST_PROTOCOL)

    plt.figure(figsize=[8,6])
    for fun in args.hidden_act_fun_list:
        x = np.arange(len(d_train_loss[fun]))
        plt.plot(x, smooth_curve(d_train_loss[fun]), label=fun)
    plt.title('Plot of Training Loss of ' + model + ' with Different Activation Functions of The Hiddern Layer')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fn.replace('.pkl', '.png'))
    print('The plot of training loss is saved as', fn)
    