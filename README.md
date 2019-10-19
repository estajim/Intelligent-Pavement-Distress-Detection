# Intelligent-Pavement-Distress-Detection
This Repo includes my works in detection of standardized pavement distresses and all the neccasary branches as I am working on it. In this independent project, I intend to enhance automated crack/distress detection tasks in my company. We have ample of manually annotated images with labels of distress types. I am planning to build Convolutional Neural Networks on such datasets to automatically detect arial cracks such as alligator cracking, block cracking, patches, etc. Therefore, this is an object detection tasks. To start, The TnesorFlow's object detector API will be used (Described in https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85) I will share more as I progress further.




# Self Updates/notes
A branch, namely parallel, keeps the unsure progress and was updated revently on 10/16/2019 11:48 pm.

test2.py creates an input table which contains all the necessary information for the xml file with all the annotation details
create_annotation2 takes the input table and create xml annotation for each img

Problems so far:

1. The DE kml file does not give disstress information per frame. The distress collection is for the whole section. Need to find a way to decompose the distress data into frames.

2. the test2.py find the inputs for each file not image. Need to be fixed.

3. length should be measured in in/ft not lon/lat (in test2.py)

4. in test2.py, you need to make a collection of distresses and assign it to the objects

5. read_ddf_dff.py has good stuff. I tried extracting the info from ddf files but the distress coordinates they have is the projected centerline coords of where the box distresses are (not their centeroids)! So Useless. I'll go back to exported kmls and try to find a coorelation between long lats of each frame and point inside them. (test2.py and create_annotations2.py)

To build to AI model, I am following the link:https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/ with the example dataset in https://github.com/experiencor/kangaroo/tree/master/annots

