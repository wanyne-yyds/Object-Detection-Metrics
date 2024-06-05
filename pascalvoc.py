###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Feb 12th 2021                                                 #
###########################################################################################

####################################################################################################
#                                                                                                  #
# THE CURRENT VERSION WAS UPDATED WITH A VISUAL INTERFACE, INCLUDING MORE METRICS AND SUPPORTING   #
# OTHER FILE FORMATS. PLEASE ACCESS IT ACCESSED AT:                                                #
#                                                                                                  #
# https://github.com/rafaelpadilla/review_object_detection_metrics                                 #
#                                                                                                  #
# @Article{electronics10030279,                                                                    #
#     author         = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto,    #
#                       Sergio L. and da Silva, Eduardo A. B.},                                    #
#     title          = {A Comparative Analysis of Object Detection Metrics with a Companion        #
#                       Open-Source Toolkit},                                                      #
#     journal        = {Electronics},                                                              #
#     volume         = {10},                                                                       #
#     year           = {2021},                                                                     #
#     number         = {3},                                                                        #
#     article-number = {279},                                                                      #
#     url            = {https://www.mdpi.com/2079-9292/10/3/279},                                  #
#     issn           = {2079-9292},                                                                #
#     doi            = {10.3390/electronics10030279}, }                                            #
####################################################################################################

####################################################################################################
# If you use this project, please consider citing:                                                 #
#                                                                                                  #
# @INPROCEEDINGS {padillaCITE2020,                                                                 #
#    author    = {R. {Padilla} and S. L. {Netto} and E. A. B. {da Silva}},                         #
#    title     = {A Survey on Performance Metrics for Object-Detection Algorithms},                #
#    booktitle = {2020 International Conference on Systems, Signals and Image Processing (IWSSIP)},#
#    year      = {2020},                                                                           #
#    pages     = {237-242},}                                                                       #
#                                                                                                  #
# This work is published at: https://github.com/rafaelpadilla/Object-Detection-Metrics             #
####################################################################################################

import glob
import os
import shutil
import sys
import torch
from pathlib import Path

# parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
# sys.path.insert(0, parent_path)

from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat, xyxyxyxy2xywhr

# Validate formats
def ValidateFormats(argFormat):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat == 'xywhr':
        return BBFormat.XYWHR
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        assert False, 'argument %s: invalid value. It must be either \'xywh\' or \'xyrb\' or \'xywhr\'' % argFormat

# Validate coordinate types
def ValidateCoordinatesTypes(arg):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    else:
        assert False, 'argument %s: invalid value. It must be either \'abs\' or \'rel\'' % arg

def create_bounding_box(nameOfImage, idClass, x, y, w, h, 
                        r=None, coordType=None, imgSize=None, bbType=None, bbFormat=None, confidence=None):
    if r is not None:
        return BoundingBox(nameOfImage, idClass, x, y, w, h, r, 
                           typeCoordinates=coordType, imgSize=imgSize, bbType=bbType, 
                           classConfidence=confidence, format=bbFormat)
    else:
        return BoundingBox(nameOfImage, idClass, x, y, w, h, 
                           typeCoordinates=coordType, imgSize=imgSize, bbType=bbType, 
                           classConfidence=confidence, format=bbFormat)

def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []

    # Read ground truths
    os.chdir(directory)
    files = glob.glob("**/*.txt", recursive=True)
    files.sort()

    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        with open(f, "r") as fh1:
            for line in fh1:
                line = line.replace("\n", "")
                if line.replace(' ', '') == '':
                    continue
                splitLine = line.split(" ")
                idClass = splitLine[0]
                if len(splitLine) > 6:
                    box = np.array([float(x) for x in splitLine[1 if isGT else 2:]])[np.newaxis, :]
                    box = torch.Tensor(box)
                    xywhr = xyxyxyxy2xywhr(box).numpy().squeeze().tolist()     # returns (centerX, centerY, w, h, rotation)
                    x, y, w, h, r = xywhr
                    bb = create_bounding_box(nameOfImage, idClass, x, y, w, h, r, coordType, 
                                             imgSize, BBType.GroundTruth if isGT else BBType.Detected, bbFormat, None if isGT else float(splitLine[1]))
                else:
                    # idClass = int(splitLine[0]) #class
                    x = float(splitLine[1 if isGT else 2])
                    y = float(splitLine[2 if isGT else 3])
                    w = float(splitLine[3 if isGT else 4])
                    h = float(splitLine[4 if isGT else 5])
                    bb = create_bounding_box(nameOfImage, idClass, x, y, w, h, None, coordType, 
                                             imgSize, BBType.GroundTruth if isGT else BBType.Detected, bbFormat, None if isGT else float(splitLine[1]))

                allBoundingBoxes.addBoundingBox(bb)
                if idClass not in allClasses:
                    allClasses.append(idClass)
    return allBoundingBoxes, allClasses


if __name__ == '__main__':
    # Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))

    # Groundtruth folder: folder containing your ground truth bounding boxes
    gtFolder = r"E:\Code\Object-Detection-Metrics\test_dataset\Test_obb_labels_groundtruths"
    # Detection folder: folder containing your detected bounding boxes
    detFolder = r"E:\Code\Object-Detection-Metrics\test_dataset\Test_obb_labels_detections"
    
    # folder where the plots are saved
    savePath = Path(currentPath) / 'results'

    # xywh: <left> <top> <width> <height>) or xyrb: <left> <top> <right> <bottom> or xywhr: xywhr
    gt_format  = 'xywhr'
    det_format = 'xywhr'

    # IOU threshold.
    iouThreshold = 0.5

    # reference of the ground truth bounding box coordinates: absolute values ('abs') or relative to its image size ('rel')
    gtCoordinates = 'abs'
    detCoordinates = 'abs'
    # image size. Required if gtCoordinates or detCoordinates are 'rel
    imgSize = (0, 0)   # width, height

    # Show plot during execution
    showPlot = True

    # Validate formats
    gtFormat = ValidateFormats(gt_format)
    detFormat = ValidateFormats(det_format)

    # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(gtCoordinates)
    detCoordType = ValidateCoordinatesTypes(detCoordinates)
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = imgSize
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = imgSize

    # Validate savePath
    # Check if path to save results already exists and is not empty
    if os.path.isdir(savePath) and os.listdir(savePath):
        key_pressed = ''
        while key_pressed.upper() not in ['Y', 'N']:
            print(f'Folder {savePath} already exists and may contain important results.\n')
            print(f'Enter \'Y\' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!')
            print(f'Or enter \'N\' to abort and choose another folder to save the results.')
            key_pressed = input('')

        if key_pressed.upper() == 'N':
            print('Process canceled')
            sys.exit()

    # Clear folder and save results
    shutil.rmtree(str(savePath), ignore_errors=True)
    savePath.mkdir(parents=True, exist_ok=True)

    # print('iouThreshold= %f' % iouThreshold)
    # print('savePath = %s' % savePath)
    # print('gtFormat = %s' % gtFormat)
    # print('detFormat = %s' % detFormat)
    # print('gtFolder = %s' % gtFolder)
    # print('detFolder = %s' % detFolder)
    # print('gtCoordType = %s' % gtCoordType)
    # print('detCoordType = %s' % detCoordType)
    # print('showPlot %s' % showPlot)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder,
                                                    True,
                                                    gtFormat,
                                                    gtCoordType,
                                                    imgSize=imgSize)
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(detFolder,
                                                    False,
                                                    detFormat,
                                                    detCoordType,
                                                    allBoundingBoxes,
                                                    allClasses,
                                                    imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision YOLOv8 Recall curve
    detections = evaluator.PlotYOLOv8PrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        OBB=True,
        savePath=savePath,)

    for name, metrics in detections.items():
        metrics = np.array(metrics).flatten()
        print(f"Class: {name}\tP: {metrics[0]:.3f}\tR: {metrics[1]:.3f}\tmAP50: {metrics[2]:.3f}\tmAP50-95: {metrics[3]:.3f}")

    # Plot Precision x Recall curve
    # detections = evaluator.PlotPrecisionRecallCurve(
    #     allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    #     IOUThreshold=iouThreshold,  # IOU threshold
    #     method=MethodAveragePrecision.EveryPointInterpolation,
    #     showAP=True,  # Show Average Precision in the title of the plot
    #     obb=True,
    #     showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
    #     savePath=savePath,
    #     showGraphic=showPlot)
    
    # f = open(os.path.join(savePath, 'results.txt'), 'w')
    # f.write('Object Detection Metrics\n')
    # f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    # f.write('Average Precision (AP), Precision and Recall per class:')

    # # each detection is a class
    # for metricsPerClass in detections:

    #     # Get metric values per each class
    #     cl = metricsPerClass['class']
    #     ap = metricsPerClass['AP']
    #     precision = metricsPerClass['precision']
    #     recall = metricsPerClass['recall']
    #     totalPositives = metricsPerClass['total positives']
    #     total_TP = metricsPerClass['total TP']
    #     total_FP = metricsPerClass['total FP']

    #     if totalPositives > 0:
    #         validClasses = validClasses + 1
    #         acc_AP = acc_AP + ap
    #         prec = ['%.2f' % p for p in precision]
    #         rec = ['%.2f' % r for r in recall]
    #         ap_str = "{0:.2f}%".format(ap * 100)
    #         # ap_str = "{0:.4f}%".format(ap * 100)
    #         print('AP: %s (%s)' % (ap_str, cl))
    #         f.write('\n\nClass: %s' % cl)
    #         f.write('\nAP: %s' % ap_str)
    #         f.write('\nPrecision: %s' % prec)
    #         f.write('\nRecall: %s' % rec)

    # mAP = acc_AP / validClasses
    # mAP_str = "{0:.2f}%".format(mAP * 100)
    # print('mAP: %s' % mAP_str)
    # f.write('\n\n\nmAP: %s' % mAP_str)