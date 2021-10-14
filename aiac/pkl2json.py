import argparse
import os, json
import mmcv
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='./test.json',
                        help='path for the ground truth(Default: )')
    parser.add_argument('--dt', type=str,
                        default='./output.pkl',
                        help='path for the results on validation set(Default: )')
    parser.add_argument('--score_thr', type=float, default=0.0001,
                        help='path for the results on validation set(Default: )')

    args = parser.parse_args()

    save_fname = 'res2net.json'

    img_infos = mmcv.load(args.gt)['images']
    img_dets = mmcv.load(args.dt)
    assert len(img_dets) == len(img_infos)
    annotations_info = []

    for idx, img_info in enumerate(img_infos):
        name = img_info['file_name']
        if name.split('.')[0] == '01263481_05_BF_00241':
            name = name.split('.')[0] + '.tif'

        dets = img_dets[idx]
        for cls_idx, det_boxes in enumerate(dets):
            for box in det_boxes:
                bbox = box.astype(float)
                if box[-1] >= args.score_thr:
                    score = bbox[-1]
                    bbox = [float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]),
                                float(bbox[3]) - float(bbox[1])]
                    category = cls_idx + 1
                    annotations_info.append({"name": name, "category_id": category, "bbox": bbox, "score": score})

    json.dump(annotations_info, open(save_fname, 'w', encoding='utf-8'), indent=4, separators=(',', ': '))
