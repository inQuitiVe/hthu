import cv2
import numpy as np
# import matplotlib.pyplot as plt
import argparse
import os

def align_images(image1, image2, opt):



    if opt.denoise_rate != 0:
        image1 = cv2.fastNlMeansDenoising(image1, h=opt.denoise_rate, templateWindowSize=7)
        image2 = cv2.fastNlMeansDenoising(image2, h=opt.denoise_rate, templateWindowSize=7)

    
    # 轉換為灰度圖
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用 ORB 檢測和描述特徵
    orb = cv2.ORB_create(nfeatures=300)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # 使用 BFMatcher 匹配特徵
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # 根據匹配距離排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配點
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 計算透視變換矩陣
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 應用透視變換
    h, w = image2.shape[:2]
    aligned_image1 = cv2.warpPerspective(image1, M, (w, h),  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return aligned_image1

def compare_images(image1_path, image2_path):
    # 讀取圖片
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    dn1 = cv2.fastNlMeansDenoising(image1, h=10, templateWindowSize=7)
    dn2 = cv2.fastNlMeansDenoising(image2, h=10, templateWindowSize=7)

    # 對齊圖片
    aligned_image1 = align_images(dn1, dn2)
    cv2.imshow("orig1", image1)
    # cv2.imshow("allign1", aligned_image1)
    cv2.imshow("orig2", image2)

    # 計算圖片之間的差異
    difference = cv2.absdiff(aligned_image1, dn2)

    brighten = cv2.convertScaleAbs(difference, alpha=6)

    # 將差異轉換為灰度圖片
    # gray_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # 設定閾值以顯示明顯的不同之處
    _, thresh_diff = cv2.threshold(brighten, 127, 255, cv2.THRESH_BINARY)

    # 顯示差異圖片
    cv2.imshow("Difference", difference)
    cv2.imshow("Brighten Difference", brighten)
    cv2.imshow("Threshold Difference", thresh_diff)

    # 等待按鍵輸入
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def get_feat(imgpath, mode = 1):
    img0 = cv2.imread(imgpath)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    imgdn1 = cv2.fastNlMeansDenoising(gray0, h=5, templateWindowSize=7)
    imgdn2 = cv2.fastNlMeansDenoising(gray0, h=10, templateWindowSize=7)
    # imgS = cv2.convertScaleAbs(img0, alpha=0.5, beta=128)

    orb = cv2.ORB_create(nfeatures=300)
    kp, des = orb.detectAndCompute(img0, None)
    imgKp1 = cv2.drawKeypoints(img0, kp, None)

    kpdn1, desdn1 = orb.detectAndCompute(imgdn1, None)
    imgKpdn1 = cv2.drawKeypoints(imgdn1, kpdn1, None)

    kpdn2, desdn2 = orb.detectAndCompute(imgdn2, None)
    imgKpdn2 = cv2.drawKeypoints(imgdn2, kpdn2, None)

    plt.figure(figsize=(15,6))
    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(imgKpdn1, cv2.COLOR_BGR2RGB))
    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(imgKpdn2, cv2.COLOR_BGR2RGB))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(imgKp1, cv2.COLOR_BGR2RGB))
    plt.show()

    if mode == 0:
        return (kp, des)
    if mode == 1:
        return (kpdn1, desdn1)
    if mode == 2:
        return (kpdn2, desdn2)
    
    raise


def main(opt):
    if opt.alignment:
        if not os.path.isdir(opt.save_dir+'/align'):
            os.makedirs(opt.save_dir+'/align')
        raw_pics_path =  os.listdir(opt.source_dir)
        if opt.align_mode == 0:
           align_pic = cv2.imread(opt.source_dir + '/' + raw_pics_path[0])
           cv2.imwrite(opt.save_dir + '/align/' + raw_pics_path[0], align_pic)
           height, width, channels = align_pic.shape
        else:
            base_pic = cv2.imread(opt.source_dir + '/' + raw_pics_path[0])
            cv2.imwrite(opt.save_dir + '/align/' + raw_pics_path[0], base_pic)
            height, width, channels = base_pic.shape
        for raw_pic_path in raw_pics_path[1:]:
            raw_pic = cv2.imread(opt.source_dir + '/' + raw_pic_path)
            raw_pic = cv2.resize(raw_pic, (width, height)) 
            if opt.align_mode == 0:
                align_pic = align_images(raw_pic, align_pic, opt)
            else:
                align_pic = align_images(raw_pic, base_pic, opt)
            cv2.imwrite(opt.save_dir + '/align/' + raw_pic_path, align_pic)
    
    target_dir = opt.source_dir
    if opt.denoise_rate != 0:
        if not os.path.isdir(opt.save_dir+'/denoise'):
            os.makedirs(opt.save_dir+'/denoise')
        raw_pics_path =  os.listdir(opt.source_dir)
        first_pic = cv2.imread(opt.source_dir + '/' + raw_pics_path[0])
        height, width, channels = first_pic.shape
        for raw_pic_path in raw_pics_path:
            raw_pic = cv2.imread(opt.source_dir + '/' + raw_pic_path)
            raw_pic = cv2.resize(raw_pic, (width, height)) 
            raw_pic = cv2.cvtColor(raw_pic, cv2.COLOR_BGR2GRAY)
            denoise_pic = cv2.fastNlMeansDenoising(raw_pic, h=opt.denoise_rate, templateWindowSize=7)
            cv2.imwrite(opt.save_dir + '/denoise/' + raw_pic_path, denoise_pic)
        target_dir = opt.save_dir + '/denoise/'
    
    if opt.calc_area:
        if not os.path.isdir(opt.save_dir+f'/thresh/{opt.blur_thresh}'):
            os.makedirs(opt.save_dir+f'/thresh/{opt.blur_thresh}')
        raw_pics_path =  os.listdir(target_dir)
        area_dict = {}
        for raw_pic_path in raw_pics_path:
            raw_pic = cv2.imread(target_dir + '/' + raw_pic_path)
            proc_pic = cv2.cvtColor(raw_pic, cv2.COLOR_BGR2GRAY)
            proc_pic = cv2.GaussianBlur(proc_pic, (opt.blur_thresh, opt.blur_thresh), 0)
            proc_pic = cv2.adaptiveThreshold(proc_pic, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 3)
            contours, _ = cv2.findContours(proc_pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(raw_pic, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(opt.save_dir+f'/thresh/{opt.blur_thresh}/' + raw_pic_path, raw_pic)
            area = 0
            for cnt in contours:
                area += cv2.contourArea(cnt)
            area_dict[int(raw_pic_path.split("h")[0])] = area/(proc_pic.shape[0]*proc_pic.shape[1])
            print(raw_pic_path, ' with area ratio : ', area/(proc_pic.shape[0]*proc_pic.shape[1]))


    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='datasets/loc1', help='source')  
    parser.add_argument('--save_dir', type=str, default='results/loc1', help='save directory')  
    parser.add_argument('--visualization', action='store_true', help='pop out image or not')  
    parser.add_argument('--denoise_rate', type=int, default=15, help='0 means turn off denoise, recommend 1~20') 
    parser.add_argument('--alignment', action='store_true', help='align pictures first')  
    parser.add_argument('--align_mode', type=int, default=0, help='0: with former, 1: with first')  
    parser.add_argument('--calc_area', action='store_true', help='calc area')  
    parser.add_argument('--blur_thresh', type=int, default=5, help='area threshold')  
    

    opt = parser.parse_args()
    print(opt)

    main(opt)



    