from torch import nn
from utils import Tester
from network import resnet34, resnet101

# Set Test parameters
params = Tester.TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_60.pth'  #'./models/ckpt_epoch_400_res34.pth'
params.testdata_dir = './testimg/3/'

# models
model = resnet34(pretrained=True, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
model.fc = nn.Linear(512, 10)
#model = resnet101(pretrained=False,num_classes=1000)  # batch_size=60, 1GPU Memory > 9000M
#model.fc = nn.Linear(512*4, 6)
# ture result  000 111 222 333 444 555
#
# Test
tester = Tester(model, params)
#tester.test_line()
tester.test_ros()