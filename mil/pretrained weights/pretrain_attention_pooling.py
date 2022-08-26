import torch, sys
sys.path.append("..")
from utils.ct_loader import CTLoaderTensors
from models.mil import AttentionMILPooling
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict



N_CHANS      = 512
# NET          = "resnet18"
# SKIP_SLICES  = 2
# 
# LR           = 0.001 # learning rate
# BATCH_SIZE   = 32
# EPOCHS       = 20
# LOSS         = torch.nn.MSELoss()
# 
# if torch.cuda.is_available():
#     cuda           = True
#     num_workers    = 8
#     dir            = "/media/avcstorage/gravo"
# else:
#     cuda           = False
#     num_workers    = 0
#     dir             = "../../../data/gravo"
# ct_loader = CTLoaderTensors(data_dir = dir, skip_slices = SKIP_SLICES, encoder = NET)
# 
# train, test = ct_loader.load_dataset()
# train_loader = torch.utils.data.DataLoader(train, 
#                             batch_size  = BATCH_SIZE, 
#                             num_workers = num_workers, 
#                             pin_memory  = cuda)
# test_loader  = torch.utils.data.DataLoader(test, 
#                             batch_size  = BATCH_SIZE, 
#                             num_workers = num_workers,
#                             pin_memory  = cuda)
# 
# 
attn_pooling = AttentionMILPooling(in_channels = N_CHANS)
# weights = torch.load("attention_pooling_pretrained.pt")
# new_state_dict = OrderedDict()
# for key, value in weights.items():
#     new_key = "attention." + key
#     new_state_dict[new_key] = value
# torch.save(new_state_dict, "attention_pooling_pretrained.pt")
# weights = torch.load("attention_pooling_pretrained.pt")
# attn_pooling.load_state_dict(new_state_dict)
# model = attn_pooling.attention
# model.train(True)
# 
# train_optimizer = torch.optim.Adam(model.parameters(), lr = LR)
# writer = SummaryWriter()
# for epoch in range(EPOCHS):
#     total_loss, c = 0, 0
#     print("EPOCH", epoch)
#     for batch in train_loader:
#         batch = batch["ct"]["data"].float().squeeze()
#         for brain in batch:
#             train_optimizer.zero_grad()
#             brain = brain.squeeze()
#             y_true = torch.tensor([[1/brain.shape[0]]] * brain.shape[0]).float()
#             y_pred = model(brain)
#             loss   = LOSS(y_pred, y_true)
#             loss.backward()                   # compute the loss and its gradients
#             train_optimizer.step()            # adjust learning weights
#             total_loss += float(loss)
#             c += 1
#     total_loss /= c
#     print("Loss train\t", total_loss)
#     writer.add_scalar("loss/train", total_loss, epoch)
# 
#     model.train(False)
#     total_loss,c = 0, 0
#     for batch in test_loader:
#         batch = batch["ct"]["data"].float().squeeze()
#         total_loss = 0
#         for brain in batch:
#             brain = brain.squeeze()
#             y_true = torch.tensor([[1/brain.shape[0]]] * brain.shape[0]).float()
#             y_pred = model(brain)
#             loss   = LOSS(y_pred, y_true)
#             total_loss += float(loss)
#             c += 1
#     total_loss /= c
#     writer.add_scalar("loss/test", total_loss, epoch)
#     print("Loss test\t", total_loss)
#     print()
#     model.train(True)
# torch.save(model.state_dict(), "attention_pooling_pretrained.pt")
