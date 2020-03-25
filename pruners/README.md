Pruning a Custom Network 
========================
- Pruning has been implemented for the following types of structural modules and networks:
    * Sequential Connectivity - AlexNet
    * Residuals - ResNet
    * Depth-wise convolutions - MobileNetV2 
    * Fire modules - SqueezeNet

- The following sections describe how to prune your own description of these modules/networks and extend to other modules or networks

Pruning the above structural modules / networks but with own model description file
--------------------------------------------------------------
> Note : The only restriction in terms of layers that can be used is that torch.Functional modules such as F.relu can't be used as these can't be automatically detected by running through model.named_modules(). 
Please replace these with the equivalent torch.nn instead.

- **Sequential Connectivity** / **AlexNet** no change required as long as each layer in alexnet is sequentially defined in model descriptino file as in models/cifar/alexnet.py

- **Residuals** / **ResNet** 
    * The *BasicBlock* module or the *Bottleneck* module need to be defined as a separate class. 
    * Use the **@basic_block** or **@bottleneck** decorators with arguments : 
        - lType : name of block type 
        - convs : list of names of all convolutional layers within the block (class) 
        - downsampling : list with names of downsampling layers within the block
    * Refer to models/cifar/resnet.py for example

- **Depth-wise convolutions** / **MobileNetV2**
    * The *MbConv* block needs to be defined as a separate class 
    * Use the **@mb_conv** decorator with arguments : 
        - lType : name of block type
        - convs : list of names of all convolutional layers within the block (class) 
        - downsampling : list with names of downsampling layers within the block
    * Refer to models/cifar/mobilenetv2.py for example

- **Fire modules** / **SqueezeNet**
    * The *Fire* block needs to be defined as a separate class 
    * Use the **@fire** decorator with arguments : 
        - lType : name of block type
        - convs : list of names of all convolutional layers within the block (class) 
    * Refer to models/cifar/squeezenet.py for example

Pruning a new kind of network 
-----------------------------
### Define a decorator
- Need to define a decorator for the special block in pruners/decorators.py that takes in the information required 
    * The decorator must update the dependency block class with the desired information and register the appropriate dependency calculator found in pruners/dependencies.py
    * It must also register the appropriate model writer in pruners/model_writers.py
    * It must also register the appropriate weight transfer unit in pruners/weight_transfer.py

### Create a pruner class 
- A file *[net_name].py* needs to be created in pruners/ with a class which inherits from the BasicPruning in pruners/base.py
- The __init__ needs to set the parameters *self.fileName* and *self.netName*

### Create a dependency calculator
- A dependency calculator must be created which implements all the abstract methods in the DependencyCalculator class in pruners/dependencies.py

### Create a model writer 
- The decorator registers a writer function defined in pruners/model_writers.py to the Writer class
- The function registered must implement the required functionality for writing out a new model description file, but can use the predefined writer functions for various commonly used layers which are already defined. 

### Create a weight transfer unit 
- This follows the same structure as the model writer and the relevant functions are found in pruners/weight_transfer.py
- Some backbone functions such as "residual_backbone" and "split_and_aggregate_backbone" are provided for residuals and concatenation based modules such as Fire modules that can be used. 

Once these are defined, the **setup_pruner** function in app.py can be updated to initialise the new pruner. 

