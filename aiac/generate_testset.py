import os
from os.path import sep, join
import json
from tqdm import tqdm
import cv2
import shutil


def findAllFilesWithSpecifiedSuffix(target_dir, target_suffix="jpg"):
    find_res = []
    target_suffix_dot = "." + target_suffix
    walk_generator = os.walk(target_dir)
    for root_path, dirs, files in walk_generator:
        if len(files) < 1:
            continue
        for file in files:
            file_name, suffix_name = os.path.splitext(file)
            if suffix_name == target_suffix_dot:
                find_res.append(os.path.join(root_path, file))
    return find_res


def init_images_info(image_path_list, replace_dir):
    img_id = 0
    images_info = []
    for image_path in tqdm(image_path_list):
        img = cv2.imread(image_path)
        w, h = img.shape[1], img.shape[0]
        image = {}
        image['width'] = w
        image['height'] = h
        image['id'] = img_id
        image['file_name'] = image_path.replace(replace_dir, '')
        # print(image['file_name'])
        images_info.append(image)
        img_id += 1

    return images_info


def init_categories_info(classname_to_id):
    categories_info = []
    for k, v in classname_to_id.items():
        category = {}
        category['id'] = v
        category['name'] = k
        categories_info.append(category)
    return categories_info


def to_coco(images_info, categories_info, annotations_info, save_fname, extra_info=None):
    instance = {}
    instance['info'] = 'willer created'
    instance['license'] = ['license']
    instance['images'] = images_info
    instance['annotations'] = annotations_info
    instance['categories'] = categories_info
    instance['extra'] = extra_info
    json.dump(instance, open(save_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def convert_test_set(target_dir, save_fname, train_ann_json):
    train_ann_json = json.load(open(train_ann_json))
    categories_info = train_ann_json['categories']
    annotations_info = []
    find_res = findAllFilesWithSpecifiedSuffix(target_dir)
    images_info = init_images_info(find_res, target_dir)
    to_coco(images_info, categories_info, annotations_info, save_fname)
    return save_fname


def main():
    target_dir = ''
    save_fname = ''

    categories_info = [{'id': 1, 'name': '氧化铁皮', 'supercategory': '氧化铁皮'},
                       {'id': 2, 'name': '裂纹系翘皮', 'supercategory': '裂纹系翘皮'},
                       {'id': 3, 'name': '精轧周期压痕', 'supercategory': '精轧周期压痕'},
                       {'id': 4, 'name': '保护渣系夹渣', 'supercategory': '保护渣系夹渣'},
                       {'id': 5, 'name': '铁皮灰', 'supercategory': '铁皮灰'}]
    annotations_info = []
    find_res = findAllFilesWithSpecifiedSuffix(target_dir)
    images_info = init_images_info(find_res, target_dir)
    to_coco(images_info, categories_info, annotations_info, save_fname)


if __name__ == "__main__":
    main()
