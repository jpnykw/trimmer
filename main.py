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

# しきい値（調節する必要はある）
t = 990

# TODO: 現状は単体のファイルだけを見るが将来的には
#       ディレクトリ直下にある画像全部に対して処理を行えるようにする
path = './images/'
file_name = '7_8hd5o4zzec'
image = cv2.imread('{}test/{}.jpg'.format(path, file_name))
height, width = image.shape[:2]

print('step =', step, ', width =', width, ', height =', height)
print('analysis start...')

for x in range(0, width, step):
    # 縦の列を区切る
    trimmed_image = image[0 : height, x : x + step]
    start_point = (x + round(step / 2), 0)
    end_point = (x + round(step / 2), height)

    plt.cla()
    plt.clf()
    level_sum = 0

    # 区切った区間に対してヒストグラムを生成
    for i, col in enumerate(('b', 'g', 'r')):
        hist = cv2.calcHist([trimmed_image], [i], None, [256], [0, 256])
        color_level = 3072.0 - np.amax(hist)
        level_sum += color_level

        plt.plot(hist, color = col)
        plt.xlim([0, 256])

        if debug:
            print(col, color_level)

    if debug:
        print('x', x, 'sum', level_sum)

    # TODO: しきい値だけで判断しているが、白の割合なども考慮して判定精度を上げたい
    if (not crop_flag and t < level_sum) or (crop_flag and t > level_sum):
        crop_flag = not crop_flag

        if crop_flag:
            crop_start_x = x
        else:
            person_count += 1
            person = image[0 : height, crop_start_x : x]
            cv2.imwrite('{}{}_{}.jpg'.format(path, file_name, person_count), person)

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

