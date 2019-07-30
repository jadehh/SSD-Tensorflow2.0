import numpy as np
bboxes = []
for i in range(4):
    x1 = 1
    y1 = 2
    x2 = 1
    y2 = 2
    bboxes.append(np.array([x1,x2,y1,y2]))

bboxes = np.array(bboxes)
x1 = bboxes[:,0]
print (bboxes)
