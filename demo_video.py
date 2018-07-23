import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torchcv.models.fpnssd import FPNSSD512, FPNSSDBoxCoder
import pdb
import sys
import cv2
import numpy as np 
import time
# id = int(sys.argv[1])

print('Loading model..')
net = FPNSSD512(num_classes=4).to('cuda')
net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load('./log_1024/net.pth')['net'])
net.eval()

size = (1242, 375)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('detection.avi', fourcc, 10.0, size)
time_list = []
# for i in range(440):
for i in range(400):
	print i

	# print('Loading image..')
	img_ori = Image.open('/data/kitti/raw/2011_09_26/2011_09_26_drive_0009_sync/image_02/data/%010d.png' %(i))
	img_ori = Image.open('/data/kitti/raw/2011_09_26/2011_09_26_drive_0095_sync/image_02/data/%010d.png' %(i))
	# img_ori = Image.open('/data/kitti/raw/2011_09_30/2011_09_30_drive_0018_sync/image_02/data/%010d.png' %(i))

	# print img_ori.size
	# size_ori = img.size
	ow = 1024
	oh = 256
	factor_x = img_ori.size[0]/float(ow)
	factor_y = img_ori.size[1]/float(oh)

	img = img_ori.copy().resize((ow,oh))
	# print img_ori.size
	# print('Predicting..')
	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
	x = transform(img)
	time1 = time.time()
	loc_preds, cls_preds = net(x.unsqueeze(0))
	time2 = time.time()

	time_list.append(time2-time1)
	# sprint('Decoding..')
	box_coder = FPNSSDBoxCoder()
	loc_preds = loc_preds.squeeze().cpu()
	cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
	boxes, labels, scores = box_coder.decode(loc_preds, cls_preds, score_thresh=0.2, nms_thresh=0.3)
	print labels
	# print img_ori.size

	draw = ImageDraw.Draw(img_ori)
	# pdb.set_trace()
	imm  = np.array(img_ori)[:,:,[2,1,0]]

	image = imm.copy()
	# pdb.set_trace()
	if not len(boxes) == 0:
		boxes[:, 0] *= factor_x
		boxes[:, 2] *= factor_x
		boxes[:, 1] *= factor_y
		boxes[:, 3] *= factor_y
		for idx, box in enumerate(boxes):
			if labels[idx] == 0:
				color = (0,255,0)
			elif labels[idx] == 1:
				color = (255,0,0)
			else:
				color = (0,0,255)
			#draw.rectangle(list(box), outline='red')
			image = cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color,2)
	# img_ori.show()
	# image  = np.array(img_ori)[:,:,[2,1,0]]
	vid.write(image)
	# pdb.set_trace()
vid.release()
time_list = np.array(time_list)[200:]
print time_list.mean()