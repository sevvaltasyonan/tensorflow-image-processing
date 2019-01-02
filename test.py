# test.py

import os
import tensorflow as tf
import numpy as np
import cv2

RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/test_images"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

#######################################################################################################################
def main():
    print("starting program . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    classifications = []
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        classification = currentLine.rstrip()
        classifications.append(classification)
    # end for

    print("classifications = " + str(classifications))

    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(retrainedGraphFile.read())
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    if not os.path.isdir(TEST_IMAGES_DIR):
        print("the test image directory does not seem to be a valid directory, check file / directory paths")
        return
    # end if

    with tf.Session() as sess:
        for fileName in os.listdir(TEST_IMAGES_DIR):
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue
            # end if

            print(fileName)

            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            openCVImage = cv2.imread(imageFileWithPath)

            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue
            # end if

            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            tfImage = np.array(openCVImage)[:, :, 0:3]
            
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            print("---------------------------------------")

            onMostLikelyPrediction = True
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]
                # end if

                confidence = predictions[0][prediction]

                if onMostLikelyPrediction:
                    scoreAsAPercent = confidence * 100.0
                    print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    writeResultOnImage(openCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    cv2.imshow(fileName, openCVImage)
                    onMostLikelyPrediction = False
                # end if

                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")
            # end for

            cv2.waitKey()
            cv2.destroyAllWindows()
        # end for
    # end with

    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False
    # end if

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    fontScale = 1.0
    fontThickness = 2

    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
