import mxnet as mx

def RSME(labels, preds):
        return np.sqrt(np.mean((labels- preds.clip(0, 1)) ** 2))

def Jaccard_Loss(labels, preds):
    intersection = np.sum(np.multiply(labels, preds))
    union = np.sum(labels**2)+np.sum(preds**2)-intersection
    return 1-intersection/float(union)

def Dice_Loss(labels, preds):
    def dice_coef(y_true, y_pred, smooth = 1.):
        intersection = np.sum(np.multiply(labels, preds))
        return (2.*intersection+smooth) / (np.sum(labels)+np.sum(preds)+smooth)

    return -1*dice_coef(labels, preds)

class SegmentationLoss(mx.metric.EvalMetric):
    def __init__(self, loss_function="Jaccard_Loss"):
        super(SegmentationLoss, self).__init__('Segementation Loss')
        self.epoch = 0
        self.loss_function = loss_function

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            assert label.shape == pred.shape
            label_num = label.asnumpy()
            pred_num = pred.asnumpy()
            self.sum_metric += eval(self.loss_function)(label_num, pred_num)
        self.num_inst += 1