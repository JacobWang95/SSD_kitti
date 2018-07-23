import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder
import pdb
import sys

id = int(sys.argv[1])

print('Loading model..')
net = FPNSSD512(num_classes=2).to('cuda')
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('/data/3dd/torchcv_rl/log1/net.pth')['net'])
net.eval()

print('Loading image..')
img_ori = Image.open('/data/kitti/3dd/training/image_2/%06d.png' %(id))
print img_ori.size
# size_ori = img.size
ow = 1024
oh = 256
factor_x = img_ori.size[0]/float(ow)
factor_y = img_ori.size[1]/float(oh)

img = img_ori.copy().resize((ow,oh))
print img_ori.size
print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
x = transform(img)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = FPNSSDBoxCoder()
loc_preds = loc_preds.squeeze().cpu()
cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
boxes, labels, scores = box_coder.decode(loc_preds, cls_preds, score_thresh=0.6, nms_thresh=0.45)
print img_ori.size

draw = ImageDraw.Draw(img_ori)
# pdb.set_trace()
if not len(boxes) == 0:
	boxes[:, 0] *= factor_x
	boxes[:, 2] *= factor_x
	boxes[:, 1] *= factor_y
	boxes[:, 3] *= factor_y
	for box in boxes:

	    draw.rectangle(list(box), outline='red')
img_ori.show()