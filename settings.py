from datetime import datetime

base_architecture = 'resnet18'
img_size = 448
prototype_shape = (2000, 128, 1, 1)
num_classes = 200
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '003'

prefix = './'
data_path = '/data/pwojcik/mito_work/dataset_512_protopool/'
tensorboard_path = prefix + '/tensorboard_'+ base_architecture +'/baseline_' + datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
# data_path = '/Users/bartoszzielinski/Databases/birds/'
# tensorboard_path = '/Users/bartoszzielinski/Code/PrototypeNet_results/birds/tensorboard/joining_1__' + datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
train_dir = data_path + '/train_cropped_augmented/'
test_dir = data_path + '/test/'
train_push_dir = data_path + '/train/'
train_batch_size = 60
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 30
num_warm_epochs = 5

push_start = 10
when_push = 5
push_epochs = [push_start + i for i in range(num_train_epochs) if i % when_push == 0]
take_best_prototypes=True
normalize_prototypes=True
proto_percnetile = 0.1
trained_model_path =''# './saved_models/resnet34/003/21_push0.7853.pth' #'./saved_models/vgg19/003/16nopush0.7620.pth'
trained_optim_path =''# './21_push0.7853_optims.pth' #'./16nopush0.7620_optims.pth'
start_epoch = 0
anneal_lr=False
reset_optim = False
share = True
l2_prune = False
