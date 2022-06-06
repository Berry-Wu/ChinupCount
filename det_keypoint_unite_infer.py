# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import cv2
import math
import numpy as np
import paddle
import yaml

import time
from common import Triangle, Point, Get_angle

from det_keypoint_unite_utils import argsparser
from preprocess import decode_image
from infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images, bench_log
from keypoint_infer import KeyPointDetector, PredictConfig_KeyPoint
from visualize import visualize_pose
from benchmark_utils import PaddleInferBenchmark
from utils import get_current_memory_mb
from keypoint_postprocess import translate_to_ori_images

KEYPOINT_SUPPORT_MODELS = {
    'HigherHRNet': 'keypoint_bottomup',
    'HRNet': 'keypoint_topdown'
}


def predict_with_given_det(image, det_res, keypoint_detector,
                           keypoint_batch_size, run_benchmark):
    rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
        image, det_res)
    keypoint_vector = []
    score_vector = []

    rect_vector = det_rects
    keypoint_results = keypoint_detector.predict_image(
        rec_images, run_benchmark, repeats=10, visual=False)
    keypoint_vector, score_vector = translate_to_ori_images(keypoint_results,
                                                            np.array(records))
    keypoint_res = {}
    keypoint_res['keypoint'] = [
        keypoint_vector.tolist(), score_vector.tolist()
    ] if len(keypoint_vector) > 0 else [[], []]
    keypoint_res['bbox'] = rect_vector
    return keypoint_res


def topdown_unite_predict(detector,
                          topdown_keypoint_detector,
                          image_list,
                          keypoint_batch_size=1,
                          save_res=False):
    det_timer = detector.get_timer()
    store_res = []
    for i, img_file in enumerate(image_list):
        # Decode image in advance in det + pose prediction
        det_timer.preprocess_time_s.start()
        image, _ = decode_image(img_file, {})
        det_timer.preprocess_time_s.end()

        if FLAGS.run_benchmark:
            results = detector.predict_image(
                [image], run_benchmark=True, repeats=10)

            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
        else:
            results = detector.predict_image([image], visual=False)
        results = detector.filter_box(results, FLAGS.det_threshold)
        if results['boxes_num'] > 0:
            keypoint_res = predict_with_given_det(
                image, results, topdown_keypoint_detector, keypoint_batch_size,
                FLAGS.run_benchmark)

            if save_res:
                save_name = img_file if isinstance(img_file, str) else i
                store_res.append([
                    save_name, keypoint_res['bbox'],
                    [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
                ])
        else:
            results["keypoint"] = [[], []]
            keypoint_res = results
        if FLAGS.run_benchmark:
            cm, gm, gu = get_current_memory_mb()
            topdown_keypoint_detector.cpu_mem += cm
            topdown_keypoint_detector.gpu_mem += gm
            topdown_keypoint_detector.gpu_util += gu
        else:
            if not os.path.exists(FLAGS.output_dir):
                os.makedirs(FLAGS.output_dir)
            visualize_pose(
                img_file,
                keypoint_res,
                visual_thresh=FLAGS.keypoint_threshold,
                save_dir=FLAGS.output_dir)
    if save_res:
        """
        1) store_res: a list of image_data
        2) image_data: [imageid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_image_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def topdown_unite_predict_video(detector,
                                topdown_keypoint_detector,
                                camera_id,
                                keypoint_batch_size=1,
                                save_res=False):
    video_name = 'output.mp4'
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        # capture = cv2.VideoCapture('/home/shine/Desktop/mmpose-master/demo/chinup.mp4')
        video_name = os.path.split(FLAGS.video_file)[-1]
    # Get Video info : resolution, fps, frame count
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("fps: %d, frame_count: %d" % (fps, frame_count))

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    # writer = cv2.VideoWriter("/home/shine/PaddleDetection/output/out.mp4", fourcc, fps, (width, height))
    index = 0
    store_res = []

    direction = 0  # 判断方向
    chinUpCnt = 0

    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        index += 1
        # print('detect frame: %d' % (index))

        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detector.predict_image([frame2], visual=False)
        results = detector.filter_box(results, FLAGS.det_threshold)
        if results['boxes_num'] == 0:
            writer.write(frame)
            continue

        keypoint_res = predict_with_given_det(
            frame2, results, topdown_keypoint_detector, keypoint_batch_size,
            FLAGS.run_benchmark)

        pose = keypoint_res['keypoint']
        pose = np.array(pose[0])

        # print(pose)
        HEAD_x = pose[0, 0, 0]
        HEAD_y = pose[0, 0, 1]

        LEFT_ANKLE_x = pose[0, 15, 0]
        LEFT_ANKLE_y = pose[0, 15, 1]
        LEFT_KNEE_x = pose[0, 13, 0]
        LEFT_KNEE_y = pose[0, 13, 1]
        LEFT_HIP_x = pose[0, 11, 0]
        LEFT_HIP_y = pose[0, 11, 1]
        RIGHT_ANKLE_x = pose[0, 16, 0]
        RIGHT_ANKLE_y = pose[0, 16, 1]
        RIGHT_KNEE_x = pose[0, 14, 0]
        RIGHT_KNEE_y = pose[0, 14, 1]
        RIGHT_HIP_x = pose[0, 12, 0]
        RIGHT_HIP_y = pose[0, 12, 1]

        LEFT_WRIST_x = pose[0, 9, 0]
        LEFT_WRIST_y = pose[0, 9, 1]
        LEFT_ELBOW_x = pose[0, 7, 0]
        LEFT_ELBOW_y = pose[0, 7, 1]
        LEFT_SHOULDER_x = pose[0, 5, 0]
        LEFT_SHOULDER_y = pose[0, 5, 1]
        RIGHT_WRIST_x = pose[0, 10, 0]
        RIGHT_WRIST_y = pose[0, 10, 1]
        RIGHT_ELBOW_x = pose[0, 8, 0]
        RIGHT_ELBOW_y = pose[0, 8, 1]
        RIGHT_SHOULDER_x = pose[0, 6, 0]
        RIGHT_SHOULDER_y = pose[0, 6, 1]

        LEFT_ELBOW_t = Triangle(Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y),
                                Point(LEFT_ELBOW_x, LEFT_ELBOW_y),
                                Point(LEFT_WRIST_x, LEFT_WRIST_y))
        RIGHT_ELBOW_t = Triangle(Point(RIGHT_SHOULDER_x, RIGHT_SHOULDER_y),
                                 Point(RIGHT_ELBOW_x, RIGHT_ELBOW_y),
                                 Point(RIGHT_WRIST_x, RIGHT_WRIST_y))
        SHOULDER_t = Triangle(Point(LEFT_ELBOW_x, LEFT_ELBOW_y),
                              Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y),
                              Point(LEFT_HIP_x, LEFT_HIP_y))

        xArray = np.array([LEFT_SHOULDER_x, LEFT_HIP_x, LEFT_KNEE_x])
        # 求肩膀，腰和膝盖的标准偏差，越小越偏向于直立
        xStdInt = np.std(xArray)

        threshold_x = 70

        if (float(xStdInt) < threshold_x) & (LEFT_WRIST_y < LEFT_ELBOW_y):

            # 两臂弯曲程度
            elbow_angle = LEFT_ELBOW_t.angle_p2()

            # 两个先验角度35~170
            if elbow_angle >= 170:
                if direction == 0:
                    chinUpCnt = chinUpCnt + 0.5
                    direction = 1
            if elbow_angle <= 35:
                if direction == 1:
                    if HEAD_y < LEFT_WRIST_y:
                        chinUpCnt = chinUpCnt + 0.5
                        direction = 0
                        print("chinUpCnt:" + str(int(chinUpCnt)))

            '''
            # 肩膀弯曲角度
            shoulder = SHOULDER_t.angle_p2()
            print("shoulder_before", shoulder_before, "shoulder", shoulder, "frameCnt_ChinUp", frameCnt_ChinUp)
            if shoulder is not None:
                if ((shoulder > 90) & (shoulder_before < 90)) | ((shoulder < 90) & (shoulder_before > 90)):
                    frameCnt_ChinUp += 1
                    # 防止抖动影响
                    if frameCnt_ChinUp < threshold_wave:
                        chinUpCnt = chinUpCnt
                    else:
                        chinUpCnt += 1
                        frameCnt_ChinUp = 0
                # 反转上一帧角度
                if shoulder_before == 180:
                    shoulder_before = 0
                else:
                    shoulder_before = 180
                print('CHINCOUNT:', chinUpCnt)
            else:
                frameCnt_ChinUp = 0
        else:
            frameCnt_ChinUp = 0
            '''

        im = visualize_pose(
            frame,
            keypoint_res,
            visual_thresh=FLAGS.keypoint_threshold,
            returnimg=True)
        if save_res:
            store_res.append([
                index, keypoint_res['bbox'],
                [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
            ])

        # writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # present the video and count
        frame = cv2.resize(im, (720, 480))
        cv2.putText(frame, "chinupcount:" + str(int(chinUpCnt)) + " frame:" + str(index), (20, 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.namedWindow('chinup')
        cv2.imshow("chinup", frame)
        keycode = cv2.waitKey(1)
        if keycode == 27:
            cv2.destroyWindow('chinup')
            videoCapture.release()
            break
        writer.write(frame)
    writer.release()
    print('output_video saved to: {}'.format(out_path))
    if save_res:
        """
        1) store_res: a list of frame_data
        2) frame_data: [frameid, rects, [keypoints, scores]]
        3) rects: list of rect [xmin, ymin, xmax, ymax]
        4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
        5) scores: mean of all joint conf
        """
        with open("det_keypoint_unite_video_results.json", 'w') as wf:
            json.dump(store_res, wf, indent=4)


def main():
    deploy_file = os.path.join(FLAGS.det_model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    detector_func = 'Detector'
    if arch == 'PicoDet':
        detector_func = 'DetectorPicoDet'

    detector = eval(detector_func)(FLAGS.det_model_dir,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn,
                                   threshold=FLAGS.det_threshold)

    topdown_keypoint_detector = KeyPointDetector(
        FLAGS.keypoint_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.keypoint_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        use_dark=FLAGS.use_dark)
    keypoint_arch = topdown_keypoint_detector.pred_config.arch
    assert KEYPOINT_SUPPORT_MODELS[
               keypoint_arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                    FLAGS.camera_id, FLAGS.keypoint_batch_size,
                                    FLAGS.save_res)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        topdown_unite_predict(detector, topdown_keypoint_detector, img_list,
                              FLAGS.keypoint_batch_size, FLAGS.save_res)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            topdown_keypoint_detector.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.det_model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')
            keypoint_model_dir = FLAGS.keypoint_model_dir
            keypoint_model_info = {
                'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                      FLAGS.keypoint_batch_size, 'KeyPoint')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2 - t1)
