import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 注意: debug を有効にする場合は出力先のディレクトリを作成してください
# (デフォルトでは ./images/ と ./images/test/ が必要)
debug = False

# この辺の定数は必ずこれらの値で初期化
crop_flag = False
crop_start_x = 0
person_count = 0

# 分割数（数値を下げるほど細分化するため処理が遅くなる）
step = 4

# samples 列のヒストグラムの平均を使用する
samples = 5

# TODO: ユーザーからの入力を受け取ってパスを設定する
if len(sys.argv) != 3:
    print('データと出力先のパスを設定してください')
    exit()

target_path = sys.argv[1]
output_path = sys.argv[2]
target_files = os.listdir(target_path)
os.listdir(output_path) # パスの存在確認

if target_path[len(target_path) - 1] != '/':
    target_path = '{}/'.format(target_path)

if output_path[len(output_path) - 1] != '/':
    output_path = '{}/'.format(output_path)

print('analysis start...')

length = len(target_files)
for file_id in range(0, length):
    split = target_files[file_id].split('.')

    if split[1] in ['png', 'jpg']:
        file_name = split[0]
        file_type = split[1]
        print('checking {}'.format(file_name))

        if file_type == 'png':
            base_image = cv2.imread('{}{}.{}'.format(target_path, file_name, file_type), -1)
            height, width = base_image.shape[:2]
            # 背景が透過されている場合は白くする
            index = np.where(base_image[:, :, 3] == 0)
            base_image[index] = [255, 255, 255, 255]

            margin_width = 0
            image = base_image.copy()

        if file_type == 'jpg':
            base_image = cv2.imread('{}{}.{}'.format(target_path, file_name, file_type))
            height, width = base_image.shape[:2]

            margin_width = 100
            margin = np.ones((height, margin_width, 3), np.uint8) * 255
            image = cv2.hconcat([margin, base_image,margin])

        # しきい値を計算
        threshold = height * 1.23

        #リスト内包表記で一気に画像を切り刻む
        trimmed_images  = [image[0 : height, x : x + step] for x in range(0, width + margin_width, step)]

        #リスト内包表記で一気にヒストグラムを計算
        hists = np.array([[cv2.calcHist([trimmed_image], [0], None, [256], [0, 256]) for i in range(3)] for trimmed_image in trimmed_images ])

        color_levels = (height * 3 - hists.max(axis = 2)).sum(axis = 1)
        color_levels = np.concatenate([np.ones((samples - (color_levels.shape[0] % samples),1)) * color_levels[0][0], color_levels])
        color_levels = color_levels.reshape(-1)
        color_levels = np.convolve(color_levels, np.ones(samples) / samples)[samples:] #移動平均を畳み込みにより計算
        higher_than_threshold = np.where(color_levels > threshold)[0].tolist()

        start_x = higher_than_threshold[0] * step
        for i in range(len(higher_than_threshold)):
            if (higher_than_threshold[i - 1] + 1 != higher_than_threshold[i]  and i != 0) or (i == len(higher_than_threshold) - 1):
                person_count += 1
                end_x = (higher_than_threshold[i - 1] + 1) * step
                person = image[0 : height, start_x : end_x]
                cv2.imwrite('{}{}_{}.{}'.format(output_path, file_name, person_count, file_type), person)
                start_x = higher_than_threshold[i] * step

        print('step =', step, ', t=', threshold, ', width =', width, ', height =', height)

print('analysis finished!')