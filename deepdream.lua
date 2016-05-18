require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'hdf5'
require 'optim'
require 'gnuplot' --or 'image'
require 'image'

--torch.setdefaulttensortype('torch.FloatTensor')

local utils = require('utils')
local dream = require('deep_dream')

-- load and preprocess the data
data_path = '/mnt/dataset/data.h5'
local dset = utils.load_data(data_path)
local X_mean
dset, X_mean = utils.preprocess_data(dset)
print('x_meann', X_mean)

-- load the pre-trained model
local model = torch.load('../aug_deep_model.bin')--:float()
--print(model)

-- pick an image:
local X = dset.X_train[8]--:float()
local y = dset.y_train[8]
local layer = 4

local kwargs = {}
kwargs.num_iterations = 100
kwargs.learning_rate = 5.0
kwargs.max_jitter = 16
kwargs.show_every = 25

local deep_img = dream.deepdream(X, X_mean, layer, model, kwargs)
image.save('deep_dream.png', image.display(deep_img))

