import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 注意: debug を有効にする場合は出力先のディレクトリを作成してください
# (デフォルトでは ./images/ と ./images/plots/ が必要)
debug = False

# 分割数（数値を下げるほど細分化するため処理が遅くなる）
step = 4

# samples 列のヒストグラムの平均を使用する
samples = 5

# 検知位置からサンプル量に合わせて左側にずらすピクセル数
diff_x = step * samples

# 特徴量を読み込んで分類器を作成
classifier = cv2.CascadeClassifier('./lbpcascade_animeface.xml')

def calc_hist_levels(image, ceil):
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    hist_r = cv2.calcHist([r], [0], None, [256], [0,256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0,256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0,256])
    return sum([ceil - hist.max() for hist in [hist_r, hist_g, hist_b]])

# TODO: ユーザーからの入力を受け取ってパスを設定する
if len(sys.argv) != 3:
    print('Set the path of the input/output image source')
    exit()

target_path = sys.argv[1]
output_path = sys.argv[2]
target_files = os.listdir(target_path)
os.listdir(output_path) # パスの存在確認

if target_path[len(target_path) - 1] != '/':
    target_path = '{}/'.format(target_path)

if output_path[len(output_path) - 1] != '/':
    output_path = '{}/'.format(output_path)

length = len(target_files)
print('Analysis start...')

for file_id in range(0, length):
    split = target_files[file_id].split('.')
    if len(split) < 2: continue
    file_name = split[0]
    file_type = split[1]

    if split[1] in ['png', 'jpg']:
        print('Checking {}...'.format(file_name))
        # 既にその画像を解析していたらスキップする
        if os.path.exists('{}{}_{}.{}'.format(output_path, file_name, 1, file_type)): continue

        if file_type == 'png':
            base_image = cv2.imread('{}{}.{}'.format(target_path, file_name, file_type), -1)
            height, width = base_image.shape[:2]
            # 背景が透過されている場合は白くする
            index = np.where(base_image[:, :, 3] == 0)
            base_image[index] = [255, 255, 255, 255]

            # 透過箇所を白塗りした png を一時的に jpg 変換して保存
            cv2.imwrite('tmp.jpg', base_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            base_image = cv2.imread('tmp.jpg')
            # 一時的に作成しただけなので削除する
            os.remove('tmp.jpg')

        if file_type == 'jpg':
            base_image = cv2.imread('{}{}.{}'.format(target_path, file_name, file_type))
            height, width = base_image.shape[:2]

        margin_width = 50
        margin_height = 50

        h_margin = np.ones((height, margin_width, 3), np.uint8) * 255 # 左右の余白
        image = cv2.hconcat([h_margin, base_image]) # 左に余白を追加
        image = cv2.hconcat([image, h_margin]) # 右に余白を追加

        v_margin = np.ones((margin_height, width + margin_width * 2, 3), np.uint8) * 255 # 拡張したサイズを計算しないと行列サイズが合わない
        image = cv2.vconcat([v_margin, image]) # 上に余白を追加
        image = cv2.vconcat([image, v_margin]) # 下に余白を追加

        # print(width, height)
        width = width + margin_width * 2
        height = height + margin_height * 2
        # print(width, height)

        # しきい値を計算
        ht = width * 0.05

        # ヒストグラムのレベルを保存する
        horizontal_levels = []
        vertical_levels = []

        # 取りうるレベルの最大値
        ceil = height * 3

        # これらの初期値は固定
        crop_flag = False
        person_count = 0
        crop_start_y = 0

        print('\033[32m', 'step =', step, ', (h)t =', ht, ', width =', width, ', height =', height, '\033[0m')

        # 複数列になっている場合があるので配列にする
        images = []

        # 複数列になっている場合を想定して上から横方向にスライスして同様にキャラの範囲を判定
        for y in range(0, height, step):
            # 水平方向
            trimmed_image = image[y : y + step, 0 : width]
            horizontal_levels.append(calc_hist_levels(trimmed_image, width * 3))

            if len(horizontal_levels) > samples:
                horizontal_levels.pop(0)
                level_sum = sum(horizontal_levels) / samples

                if (not crop_flag and ht < level_sum) or (crop_flag and ht > level_sum):
                    crop_flag = not crop_flag

                    if crop_flag:
                        crop_start_y = y - step * samples
                    else:
                        if width / (y - crop_start_y) > 10: continue
                        line_image = image[crop_start_y - diff_x : y, 0 : width]
                        images.append(line_image)

        print('\033[32m', 'vertical images', len(images), '\033[0m')

        # 列を分離したあとで縦方向に処理していく
        for image in images:
            crop_flag = False
            crop_start_x = 0

            # サイズとしきい値を再計算
            height, width = image.shape[:2]
            width += margin_width * 2 # 左右に余白がある分を加算
            height += margin_height * 2 # 上下に余白がある分を加算
            vt = height * 1.23 # キャラクターの境界を示すしきい値
            
            print('\033[32m', 'step =', step, ', (v)t =', vt, ', width =', width, ', height =', height, '\033[0m')

            for x in range(0, width, step):
                # 垂直方向
                trimmed_image = image[0 : height, x : x + step]
                # 区切った区間に対してヒストグラムを生成
                vertical_levels.append(calc_hist_levels(trimmed_image, height * 3))

                if len(vertical_levels) > samples:
                    vertical_levels.pop(0)
                    level_sum = sum(vertical_levels) / samples

                    # TODO: しきい値だけで判断しているが、白の割合なども考慮して判定精度を上げたい
                    if (not crop_flag and vt < level_sum) or (crop_flag and vt > level_sum):
                        crop_flag = not crop_flag

                        if crop_flag:
                            crop_start_x = x
                        else:
                            # あまりに縦長の画像を弾くためにアスペクト比が 1:5 を超える画像は無視する
                            # 髪の毛などを人として誤検知してしまった場合に必要な判定
                            if height / (x - crop_start_x) > 5: continue

                            person_count += 1
                            person = image[0 : height, crop_start_x - diff_x : x]
                            new_file_path = '{}{}_{}.{}'.format(output_path, file_name, person_count, file_type)

                            # 分類器に画像を食わせて顔の位置を取得する
                            gray_image = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
                            faces = classifier.detectMultiScale(gray_image)
                            faces_count = len(faces)

                            # TODO: 二人以上の顔を検知した場合は人数に合わせて画像を分割する
                            if faces_count > 1:
                                buffer = (-1, -1)
                                split = False

                                for face_x, face_y, face_width, face_height in faces:
                                    if buffer[0] == -1:
                                        buffer = (face_x, face_x + face_width)
                                    else:
                                        # 検出した顔の範囲がかぶっている場合は誤検知なので別人とはカウントしない
                                        in_lface_x = (face_x < buffer[0]) and (buffer[0] < face_x + face_width)
                                        in_rface_x = (buffer[0] < face_x) and (face_x < buffer[1])
                                        if not (in_lface_x or in_rface_x): split = True
                                        buffer = (face_x, face_x + face_width)

                                    if debug: cv2.rectangle(person, (face_x, face_y), (face_x + face_width, face_y + face_height), (0, 0, 255), 2)

                                if split:
                                    # 別人である可能性がある場合は一人あたりの領域を人数で割って分割
                                    w_area = round((x - (crop_start_x - diff_x)) / faces_count)
                                    x_area = crop_start_x - diff_x
                                    for _ in range(0, faces_count):
                                        person_count += 1
                                        person = image[0 : height, x_area : x_area + w_area]
                                        new_file_path = '{}{}_{}.{}'.format(output_path, file_name, person_count, file_type)
                                        print(' - New person detected; save in {}'.format(new_file_path))
                                        cv2.imwrite(new_file_path, person)
                                        x_area += w_area
                            else:
                                print(' - New person detected; save in {}'.format(new_file_path))
                                cv2.imwrite(new_file_path, person)

                if debug:
                    debug_image = image.copy()
                    # cv2.rectangle(debug_image, start_point, end_point, (0, 180, 0), 1)

                    debug_width = round(width / 2)
                    debug_height = round(height / 2)

                    debug_image = cv2.resize(debug_image, None, fx = 1 / 2, fy = 1 / 2)
                    cv2.imshow('image', debug_image)
                    cv2.moveWindow('image', 800, 150)

                    plt.title('x = {} (step = {})'.format(x, step))
                    # plt.savefig('./images/plots/{}_{}.jpg'.format(x, step))
                    plt.show()
                    exit()
    else:
        print('Unsupported file types *.{}'.format(file_type))

print('Analysis finished!')

