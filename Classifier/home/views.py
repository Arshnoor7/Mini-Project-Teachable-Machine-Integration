from django.shortcuts import render
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http import JsonResponse
import time,os
from django.core.files.storage import default_storage

from django.contrib import messages
# Create your views here.



def processfunc(location):
    from keras.models import load_model  # TensorFlow is required for Keras to work
    from PIL import Image, ImageOps  # Install pillow instead of PIL
    import numpy as np

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(location).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)
    res=class_name[2:]
    return res

def home(request):
    if request.method == 'POST':

        input_data = request.FILES['formFile']
        
        # file_path=default_storage.save('Images/' + input_data.name, input_data)

        # absolute_file_path = os.path.join(default_storage.location, file_path)
    
        output_data = processfunc(input_data)

        # os.remove(absolute_file_path)

        context = {'output_data': output_data}
        return render(request, "index.html", context)

    return render(request,"index.html")

    
            
    
 
