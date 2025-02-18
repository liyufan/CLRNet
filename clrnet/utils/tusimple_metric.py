import numpy as np
from sklearn.linear_model import LinearRegression
import json as json
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85
    y_true = []
    y_pred = []

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time, pred_cls, gt_cls):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1., 0.
        angles = [
            LaneEval.get_angle(np.array(x_gts), np.array(y_samples))
            for x_gts in gt
        ]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        correct = 0
        for x_gts, thresh, cls in zip(gt, threshs, gt_cls):
            accs = [
                LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts),
                                       thresh) for x_preds in pred
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
                matched_idx = np.argmax(accs)
                LaneEval.y_true.append(cls)
                LaneEval.y_pred.append(pred_cls[matched_idx])
                if pred_cls[matched_idx] == cls:
                    correct += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return s / max(min(4.0, len(gt)),
                       1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(
                           min(len(gt), 4.), 1.), correct / matched if matched > 0 else 0.

    @staticmethod
    def bench_one_submit(
        pred_file,
        gt_file,
        cls_merge=None,
        display_labels=None,
        cm_file='confusion_matrix.svg',
    ):
        try:
            json_pred = [
                json.loads(line) for line in open(pred_file).readlines()
            ]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception(
                'We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        cls_accuracy = 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception(
                    'raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            pred_categories = pred['categories']
            if raw_file not in gts:
                raise Exception(
                    'Some raw_file from your predictions do not exist in the test tasks.'
                )
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            gt_categories = gt['categories'] if 'categories' in gt else list(map(int, gt['classes'].split(' ')))
            if cls_merge:
                gt_categories = list(map(cls_merge.get, gt_categories))
            try:
                a, p, n, cls_a = LaneEval.bench(pred_lanes, gt_lanes, y_samples,
                                         run_time, pred_categories, gt_categories)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
            cls_accuracy += cls_a
        num = len(gts)
        ConfusionMatrixDisplay.from_predictions(
            LaneEval.y_true,
            LaneEval.y_pred,
            normalize='true',
            display_labels=display_labels,
        )
        plt.savefig(cm_file)
        # the first return parameter is the default ranking parameter

        fp = fp / num
        fn = fn / num
        tp = 1 - fp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = accuracy / num
        cls_accuracy = cls_accuracy / num


        return json.dumps([{
            'name': 'Accuracy',
            'value': accuracy,
            'order': 'desc'
        }, {
            'name': 'F1_score',
            'value': f1,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': fp,
            'order': 'asc'
        }, {
            'name': 'FN',
            'value': fn,
            'order': 'asc'
        }, {
            'name': 'cls_accuracy',
            'value': cls_accuracy,
            'order': 'desc'
        }]), accuracy, cls_accuracy


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e.message)
        sys.exit(e.message)
