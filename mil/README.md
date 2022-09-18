### Experiments

 - Clip         True, False
 - Skip slices  1,2,3,4,5
 - MIL pooling  Mean, Max, Attetion
 - Encoder      ResNet18,34,50,custom_3D_cnn_v1
 <!-- EfficientNet b0,b1,b2,b3 -->
 
 
### "Normal" Training settings
 - LR          = 0.001
 - WD          = 0.001
 - STEP_SIZE   = 150
 - EPOCHS      = 300
 - Slice encoder pretrained frozen
 - Attention pooling initialized with kaiming_normal (the other poolings don't have weights)
 - Feature extractor initialized with kaiming_normal
 - MLP initialized with xavier_normal



### Attetion Pooling finetune settings

 - LR          = 0.001 # learning rate
 - WD          = 0.0025 # weight decay
 - STEP_SIZE   = 40
 - EPOCHS      = 50
 - feature extractor frozen with Mean Pooling feature extractor
 - mlp frozen with Mean Pooling mlp
 - Attention Mil Pooling initialized with kaiming_normal
 - worked well for ResNet18 Clip and skip_slices = 2
