Pruning a Custom Network 
========================
- Pruning has been implemented for the following types of structural modules and networks:
    * Sequential Connectivity - AlexNet
    * Residuals - ResNet
    * Depth-wise convolutions - MobileNetV2 
    * Fire modules - SqueezeNet

- The following sections describe how to prune your own description of these modules/networks and extend to other modules or networks

Pruning the above networks but with own model description file
--------------------------------------------------------------
- **AlexNet** no change required as long as each layer in alexnet is sequentially defined in model descriptino file as in models/cifar/alexnet.py

- **ResNet** 
    * The *BasicBlock* module or the *Bottleneck* module need to be defined as a separate class. 
    * Use the **@basic_block** or **@bottleneck** decorators with arguments : 
        - lType : name of block type 
        - convs : list of names of all convolutional layers within the block (class) 
        - downsampling : list with names of downsampling layers within the block
    * Refer to models/cifar/resnet.py for example

- **MobileNetV2**
    * The *MbConv* block needs to be defined as a separate class 
    * Use the **@mb_conv** decorator with arguments : 
        - lType : name of block type
        - convs : list of names of all convolutional layers within the block (class) 
        - downsampling : list with names of downsampling layers within the block
    * Refer to models/cifar/mobilenetv2.py for example

- **SqueezeNet**
    * The *Fire* block needs to be defined as a separate class 
    * Use the **@fire** decorator with arguments : 
        - lType : name of block type
        - convs : list of names of all convolutional layers within the block (class) 
    * Refer to models/cifar/squeezenet.py for example

Pruning a new kind of network 
-----------------------------
