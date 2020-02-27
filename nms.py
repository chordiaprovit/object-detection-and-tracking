import numpy as np

def non_max_suppression(boxes,scores,classes,cols,rows):
        assert boxes.shape[0] == scores.shape[0]
        pick = []
    # for bbox in boxes:
        x1 = boxes[:,1] * cols
        y1 = boxes[:,0] * rows
        x2 = boxes[:,3] * cols
        y2 = boxes[:,2] * rows

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indxs = np.argsort(y2)

        while len(indxs) > 0:
            last = len(indxs) - 1
            i = indxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[indxs[:last]])
            yy1 = np.maximum(y1[i], y1[indxs[:last]])
            xx2 = np.minimum(x2[i], x2[indxs[:last]])
            yy2 = np.minimum(y2[i], y2[indxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[indxs[:last]]

            indxs = np.delete(indxs, np.concatenate(([last],
                                                   np.where(overlap > 0.7)[0])))
        return boxes[pick],scores[pick],classes[pick]