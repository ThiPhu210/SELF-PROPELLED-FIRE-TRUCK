# **PROJECT: SELF-PROPELLED-FIRE-TRUCK**
## *Desciption: The self-propelled fire truck will detect and identify the fire through a USB camera, then move towards the fire and use a pump to draw water and extinguish it.*
1. Hardware Installation: You must install the circuit as Ciruit/circuit_project.jpg.
2. Training CNN Model: You must have a dataset (TRAIN: FIRE & NONFIRE and TEST: FIRE & NONFIRE). I provided you a DATASET.rar, you can extract all to ultilize it. Then you run a code Training/main.py.
3. Convert Tensorflow model to Tensorflow Lite model: After training a model CNN you comment all code, and run this paragraph belows.
### model_cnn.export('saved_model')
### converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
### tflite_model = converter.convert() 
### with open('converted_model.tflite', 'wb') as f: tf.write(tflite_model)
4. Code in raspberrypi4: You just run Raspberry_code/main.py. Remember that: in your raspberrypi4 must have the file "converted_model.tflite" which converted in step 3.
5. See the result: You can see the result in Demo/Demo.mp4.
## * NOTE: File cascade.xml must appear in all folders I provided.
