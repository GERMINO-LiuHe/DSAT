from model import FAN
from utils import *
from dataset import Dataset
import torch.utils.data as data
import hello
from Config import get_CTranS_config

config = get_CTranS_config()
# 1. initialize model and weights
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

checkpoint = './checkpoint/result.tar'
model = FAN(config, 3, 81)
state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
model.load_state_dict(state_dict, False)
model.eval()
model.to(device)

imgdirs_test_commomset = ['./data/ibug/']
testset = Dataset(imgdirs_test_commomset, 'test', 'test', 0, 128, 3)
test_loader = data.DataLoader(testset, batch_size=1, num_workers=0, pin_memory=True)
data_iter = iter(test_loader)
error = 0
error1 = 0
leng = testset.__len__()
print(leng)

NME = []

for i in range(leng):
    images, targets, kps, tforms = next(data_iter)
    img = images.to(device)
    img_part = images.squeeze(0).numpy()
    img_part = np.swapaxes(img_part, 0, 1)
    img_part = np.swapaxes(img_part, 1, 2)
    images_flip = torch.from_numpy(images.cpu().numpy()[:, :, :, ::-1].copy())  # 左右翻转
    img1 = images_flip.to(device)
    minval = 1
    with torch.no_grad():
        out, mask = model(img)
        out1, mask = model(img1)
        out1 = flip_channels(out1.cpu())
        out1 = shuffle_channels_for_horizontal_flipping(out1)
        out = (out1.cpu() + out.cpu()) / 2
        rmse, pred_pts = rmse_batch1(out, kps, tforms)
        heatmap = out[:, 0:68, :, :].detach().cpu().numpy()
        cut_size = 3
        sub_kpts = hello.get_subpixel_from_kpts(pred_pts, heatmap, cut_size)
        sub_kpts = torch.from_numpy(sub_kpts)
        sub_kpts = sub_kpts.view(-1, 68, 2)
        rmse2 = per_image_rmse(sub_kpts, kps, tforms)
        if minval > rmse:
            minval = rmse
        if math.isnan(rmse2):
            # rmse2 = 0
            rmse2 = minval
        error = error + rmse
        error1 = error1 + rmse2
        NME.append(rmse)
        print(str(i) + ' rmse is: ', rmse, rmse2)
NME.sort()
print('mean error is: ', error / leng, error1 / leng)
