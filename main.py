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
            image = cv2.hconcat([margin, base_image])

        # しきい値を計算
        t = height * 1.23

        # ヒストグラムのレベルを保存する
        levels = []

        print('step =', step, ', t=', t, ', width =', width, ', height =', height)

        for x in range(0, width + margin_width, step):
            # 縦の列を区切る
            trimmed_image = image[0 : height, x : x + step]

            if debug:
                start_point = (x + round(step / 2), 0)
                end_point = (x + round(step / 2), height)

            if debug:
                plt.cla()
                plt.clf()

            level_sum = 0

            # 区切った区間に対してヒストグラムを生成
            for i, col in enumerate(('b', 'g', 'r')):
                hist = cv2.calcHist([trimmed_image], [i], None, [256], [0, 256])
                color_level = height * 3 - np.amax(hist)
                level_sum += color_level

                if debug:
                    plt.plot(hist, color = col)
                    plt.xlim([0, 256])
                    print(col, color_level)

            if debug:
                print('x', x, 'sum', level_sum)

            levels.append(level_sum)

            if len(levels) > samples:
                levels.pop(0)
                level_sum = sum(levels) / samples

                # TODO: しきい値だけで判断しているが、白の割合なども考慮して判定精度を上げたい
                if (not crop_flag and t < level_sum) or (crop_flag and t > level_sum):
                    crop_flag = not crop_flag

                    if crop_flag:
                        crop_start_x = x
                    else:
                        if height / (x - crop_start_x) > 10:
                            continue

                        person_count += 1
                        person = image[0 : height, crop_start_x : x]
                        cv2.imwrite('{}{}_{}.{}'.format(output_path, file_name, person_count, file_type), person)
                        print('detect person (id = {})'.format(person_count))

            if debug:
                debug_image = image.copy()
                cv2.rectangle(debug_image, start_point, end_point, (0, 180, 0), 1)

                debug_width = round(width / 2)
                debug_height = round(height / 2)

                debug_image = cv2.resize(debug_image, None, fx = 1 / 2, fy = 1 / 2)
                cv2.imshow('image', debug_image)
                cv2.moveWindow('image', 800, 150)

                plt.title('x = {} (step = {})'.format(x, step))
                plt.savefig('./images/plots/{}_{}.jpg'.format(x, step))
                plt.show()

print('analysis finished!')

