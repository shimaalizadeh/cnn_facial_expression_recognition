require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'

local function reduce_net(full_net,end_layer)

    local net = nn.Sequential()
    for l=1, end_layer do
        net:add(full_net:get(l))
    end

    return net
end


local function deepdream(x, X_mean, layer, model, kwargs)
  --[[
  Generate a DeepDream image.
  
  Inputs:
  - X: Starting image, of shape (1, 3, H, W)
  - layer: Index of layer at which to dream
  - model: A PretrainedCNN object
  
  Keyword arguments:
  - learning_rate: How much to update the image at each iteration
  - max_jitter: Maximum number of pixels for jitter regularization
  - num_iterations: How many iterations to run for
  - show_every: How often to show the generated image
  ]]--
  
  local X = torch.Tensor(x:size()):copy(x)
  local H = X:size(2)
  local W = X:size(3)

  -- reduce model
  local net = reduce_net(model, layer)
  
  num_iterations = kwargs.num_iterations --100
  learning_rate = kwargs.learning_rate -- 5.0
  max_jitter = kwargs.max_jitter -- 16
  show_every = kwargs.show_every --25
  
  for t= 1, num_iterations do

    --As a regularizer, add random jitter to the image
    local ox = math.random(-max_jitter, max_jitter)
    local oy = math.random(-max_jitter, max_jitter)
    X = image.translate(X, ox, oy)
    X = X:view(1, X:size(1), H, W)
    
    --[[Compute the image gradient dX using the DeepDream method. You'll   
    need to use the forward and backward methods of the model object to    
    extract activations and set gradients for the chosen layer. After        
    computing the image gradient dX, you should use the learning rate to     
    update the image X.
   ]]--                                 

    local cuda_X = torch.CudaTensor(X:size()):copy(X:cuda())--:view(1,X:size(2),X:size(3),X:size(4))
    local layer_output = net.forward(cuda_X)
    print(cuda_X:type(), layer_output:type())                
    local dX = net.updateGradInput(cuda_X, layer_output):squeeze()

    -- apply normalized ascent to the input
    X:add(dX:mul(learning_rate / torch.abs(dX):mean()))
    --[[############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################--]]
    
    -- Undo the jitter
    X = image.translate(X, -ox, -oy)
    
    
    -- As a regularizer, clip the image
    local mean_pixel = torch.mean(X_mean)
    X:clamp( -mean_pixel, 255.0 - mean_pixel)

  end
    
  return X

end

return { reduce_net = reduce_net,
         deepdream = deepdream }

--NOTE: torch.setdefaulttensortype('torch.FloatTensor')
-- net = torch.load('./OverFeatModel.t7'):float()
