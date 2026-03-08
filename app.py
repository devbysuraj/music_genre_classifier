from flask import Flask,request,app,jsonify,url_for,render_template
import io
import librosa
from IPython.display import Audio
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

app = Flask(__name__)

classes = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
model = tf.keras.models.load_model('Trained_model.h5')

@app.route('/')
def home():
    return render_template('home.html')


def load_and_preprocess_data(file_path,target_shape = (150,150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr = None)

    #define the duration of each chunk and overlap
    chunk_duration = 4
    overlap_duration = 2
  
    #convert duration to sample
    chunk_samples = chunk_duration*sample_rate
    overlap_samples = overlap_duration*sample_rate 

    #calculate  the number of chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples)/(chunk_samples - overlap_samples))) +1
    
    # Iterate over each chunk
    for i in range(num_chunks):
        # Calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        
        # Extract the chunk of audio
        chunk = audio_data[start:end]

        # melspectrogram part, this is the matrix we are getting the 
        mel_spectrogram = librosa.feature.melspectrogram(y = chunk, sr = sample_rate) #calculating spectrogram by this

        #resizing matrix on given target shape as written in the function arguments, you can see the size of melsprectorgram
        # in the visualization part above

        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis = -1), target_shape)

        # appending the data to the lists
        data.append(mel_spectrogram)
                    
    #return with typecast to np array                
    return np.array(data)    

def model_prediction(X_test):
    y_pred = model.predict(X_test)
    normal_result = np.argmax(y_pred, axis = 1)
    unique_element, counts = np.unique(normal_result,return_counts=True)
    max_count = np.max(counts)
    return unique_element[counts == max_count][0]



# @app.route('/predict', methods=['POST'])
# def predict():
#     print("--- Prediction Started ---")
#     file = request.files.get('audio_file')
    
#     if not file:
#         return "No file received"
    
#     print(f"File received: {file.filename}")
    
#     # Read bytes
#     audio_bytes = file.read()
#     print(f"File size: {len(audio_bytes)} bytes")
    
#     audio_stream = io.BytesIO(audio_bytes)

#     try:
#         print("Starting preprocessing...")
#         X_test = load_and_preprocess_data(audio_stream)
#         print(f"Preprocessing done. Shape: {X_test.shape}")
        
#         print("Starting model prediction...")
#         prediction = model_prediction(X_test)
#         print(f"Prediction result: {prediction}")

#         return render_template("home.html", predicted_genre=f"The predicted genre is {prediction}")
    
#     except Exception as e:
#         print(f"CRASH ERROR: {str(e)}")
#         return f"Model Error: {str(e)}"


@app.route('/predict', methods = ['POST'])
def predict():
    # if 'audio_file' not in request.files:  #by the name attribute in the input tag in the form through api given in the form
    #     return "NO file uploaded", 400
    file = request.files.get('audio_file')

    if not file or file.filename == '':
        return render_template("home.html", predicted_genre="Error: No file selected")
    
# We wrap the file stream in BytesIO so Librosa treats it like a file
    audio_bytes = file.read()
    audio_stream = io.BytesIO(audio_bytes)

    try:
         
        X_test = load_and_preprocess_data(audio_stream)
        prediction = model_prediction(X_test)
        result = classes[prediction]

        return render_template("home.html", predicted_genre = "The predicted genre is {}".format(result))

    except Exception as e:
            return render_template("home.html", predicted_genre="Error processing audio: {}".format(str(e)))

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)

