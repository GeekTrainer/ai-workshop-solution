import os, base64, json, requests
from flask import Flask, render_template, request, flash

# Import Cognitive Services SDK components
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import ComputerVisionErrorException
from msrest.authentication import CognitiveServicesCredentials

# Configure system variables with dotenv
from dotenv import load_dotenv
load_dotenv()

# Create a ComputerVisionClient for calling the Computer Vision API
endpoint = os.environ["ENDPOINT"]
vision_key = os.environ["VISION_KEY"]
vision_credentials = CognitiveServicesCredentials(vision_key)
vision_client = ComputerVisionClient(endpoint, vision_credentials)

translate_key = os.environ["TRANSLATE_KEY"]

from azure.cognitiveservices.vision.face import FaceClient
face_api_key=os.environ["FACE_API_KEY"]
face_client = FaceClient(endpoint, CognitiveServicesCredentials(face_api_key))
person_group_id = 'reactor'

app = Flask(__name__)
app.secret_key = os.urandom(24)

@app.route("/", methods=["GET", "POST"])
def index():
    target_language="en"

    if request.method == "POST":
        # Display the image that was uploaded
        image = request.files["file"]
        uri = "data:image/jpg;base64," + base64.b64encode(image.read()).decode("utf-8")
        image.seek(0)

        # Use the Computer Vision API to extract text from the image
        lines = extract_text_from_image(image, vision_client)

        # Use the Translator Text API to translate text extracted from the image
        target_language = request.form["target_language"]
        translated_lines = translate_text(lines, target_language, translate_key)

        # Flash the translated text
        for translated_line in translated_lines:
            flash(translated_line)

    else:
        # Display a placeholder image
        uri = "/static/placeholder.png"

    return render_template("index.html", image_uri=uri, target_language=target_language)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'GET':
        return render_template('train.html')

    if 'file' not in request.files:
        return 'No file detected'

    image = request.files['file']
    name = request.form['name']

    # Helper function to ensure a Face API group is created
    people = get_people()

    # Look to see if the name already exists
    # If not create it
    operation = "Updated"
    person = next((p for p in people if p.name.lower() == name.lower()), None)
    if not person:
        operation = "Created"
        person = face_client.person_group_person.create(person_group_id, name)

    # Add the picture to the person
    face_client.person_group_person.add_face_from_stream(person_group_id, person.person_id, image)

    # Train the model
    face_client.person_group.train(person_group_id)

    # Display the page to the user
    return render_template('train.html', message="{} {}".format(operation, name))

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html')

    if 'file' not in request.files:
        return 'No file detected'
    image = request.files['file']

    # Find all the faces in the picture
    faces = face_client.face.detect_with_stream(image)

    # Get just the IDs so we can see who they are
    face_ids = list(map((lambda f: f.face_id), faces))

    # Ask Azure who the faces are
    identified_faces = face_client.face.identify(face_ids, person_group_id)

    names = get_names(identified_faces)

    if len(names) > 0:
        # Display the people
        return render_template('detect.html', names=names)
    else:
        # Display an error message
        return render_template('detect.html', message="Sorry, nobody looks familiar")

def get_names(identified_faces):
    names = []
    for face in identified_faces:
        # Find the top candidate for each face
        candidates = sorted(face.candidates, key=lambda c: c.confidence, reverse=True)
        # Was anyone recognized?
        if candidates and len(candidates) > 0:
            # Get just the top candidate
            top_candidate = candidates[0]
            # See who the person is
            person = face_client.person_group_person.get(person_group_id, top_candidate.person_id)

            # How certain are we this is the person?
            if top_candidate.confidence > .8:
                names.append('I see ' + person.name)
            else:
                names.append('I think I see ' + person.name)
    return names

def get_people():
    try:
        face_client.person_group.create(person_group_id, name=person_group_id)
    except:
        pass
    people = face_client.person_group_person.list(person_group_id)
    return people

def translate_text(lines, target_language, key):
    uri = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=" + target_language

    headers = {
        'Ocp-Apim-Subscription-Key': translate_key,
        'Content-type': 'application/json'
    }

    input=[]

    for line in lines:
        input.append({ "text": line })

    try:
        response = requests.post(uri, headers=headers, json=input)
        response.raise_for_status() # Raise exception if call failed
        results = response.json()

        translated_lines = []

        for result in results:
            for translated_line in result["translations"]:
                translated_lines.append(translated_line["text"])

        return translated_lines

    except requests.exceptions.HTTPError as e:
        return ["Error calling the Translator Text API: " + e.strerror]

    except Exception as e:
        return ["Error calling the Translator Text API"]

# Function that extracts text from images
def extract_text_from_image(image, client):
    try:
        result = client.recognize_printed_text_in_stream(image=image)

        lines=[]
        if len(result.regions) == 0:
            lines.append("Photo contains no text to translate")

        else:
            for line in result.regions[0].lines:
                text = " ".join([word.text for word in line.words])
                lines.append(text)

        return lines

    except ComputerVisionErrorException as e:
        return ["Computer Vision API error: " + e.message]

    except:
        return ["Error calling the Computer Vision API"]
