### Experiments

 - Clip         True, False
 - Skip slices  0,1,2,3,4,5
 - Encoder      ResNet18,34,50 EfficientNet b0,b1,b2,b3



### Attetion Pooling finetune settings

 - LR          = 0.001 # learning rate
 - WD          = 0.0025 # weight decay
 - STEP_SIZE   = 40
 - EPOCHS      = 50
 - feature extractor frozen with Mean Pooling feature extractor
 - mlp frozen with Mean Pooling mlp
 - Attention Mil Pooling initialized with kaiming_normal
 - worked well for ResNet18 Clip and skip_slices = 2
