import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

CLASS_NAMES = ['Cover', 'JMiPOD', 'UERD', 'JUNIWARD']

class Steganalyser:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyse(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            self.model.eval()
            logits = self.model(image)
            probabilities = torch.softmax(logits, dim=1).cpu().squeeze().numpy()

        class_descriptions = {
            'Cover': 'The image is not manipulated.',
            'JMiPOD': "This algorithm modifies the least detectable parts of a JPEG image to embed data, aiming to minimize changes in the image's probability distribution to avoid detection.",
            'UERD': "It uniformly distributes hidden data across all parts of a JPEG image's frequency components. This even distribution helps in avoiding patterns that could be detected by steganalysis tools",
            'JUNIWARD': "Targets the noisier, high-frequency areas of an image for data embedding, making alterations less noticeable and more difficult to detect"
        }

        class_names = ['Cover', 'JMiPOD', 'UERD', 'JUNIWARD']
        predicted_index = probabilities.argmax()
        predicted_class = class_names[predicted_index]
        description = class_descriptions[predicted_class]

        return probabilities, description


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

    model_path = 'model_multiclass.pt'
    if not os.path.isfile(model_path):
        raise RuntimeError("Model file not found. Check the path.")

    state_dict = torch.load(model_path, map_location=device)
    # Remove 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)

    return model, device


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    message = None
    image_url = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            model, device = load_model()
            analyser = Steganalyser(model, device)
            probabilities, description = analyser.analyse(file_path)

            class_names = ['Cover', 'JMiPOD', 'UERD', 'JUNIWARD']
            predicted_index = probabilities.argmax()
            predicted_class = class_names[predicted_index]

            predicted_message = f"<h3><strong>Predicted Class:</strong> <b>{predicted_class}</b></h3>"

            probabilities_text = "<ul>"
            for i, prob in enumerate(probabilities):
                class_name = class_names[i]
                prob_text = f"<li><strong>{class_name}:</strong> {prob * 100:.2f}%</li>"
                probabilities_text += prob_text
            probabilities_text += "</ul>"

            message = f"{predicted_message}{probabilities_text}<br><p><strong>Description:</strong> {description}</p>"
            image_url = url_for('uploaded_file', filename=filename)

    return render_template('index.html', message=message, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=False)

