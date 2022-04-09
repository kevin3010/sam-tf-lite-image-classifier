import base64
import json
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
from io import BytesIO

from preprocessing import preprocess_numpy_input

def lambda_handler(event, context):

    #featching image from event stored in base64 format
    image_bytes = event['body'].encode('utf-8')


    #pre-processing the image
    img = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode='RGB').resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_numpy_input(x,data_format='channels_last',mode='tf')


    #defining the tflite interpreter and running the inference 
    interpreter = tflite.Interpreter(model_path="models/mobilenet.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    output = np.squeeze(interpreter.get_tensor(output_details['index']))


    #mapping the output of the model to lable
    data = {}
    with open("models/imagenet_class_index.json") as jsonFile: 
        data = json.load(jsonFile)
    index = np.argmax(output)

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": "{0} : {1}".format(data[str(index)],output[index]),
            }
        )
    }
