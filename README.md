# 🎨 Neural Style Transfer (NST)

Transform ordinary photographs into artistic images using **Deep Learning and Neural Style Transfer**.

This project implements **Adaptive Instance Normalization (AdaIN)** with PyTorch to combine the content of an input image with the visual style of a reference image. A Flask web interface allows users to upload images and control the intensity of the style transfer.

## ✨ Features

- **Neural Style Transfer:** Apply the artistic characteristics of one image to another.
- **AdaIN:** Align the channel-wise feature statistics of content and style images.
- **VGG-19 Encoder:** Extract pretrained visual features from input images.
- **Custom Decoder:** Reconstruct stylized images from normalized feature representations.
- **Adjustable Style Intensity:** Control the balance between the original content and the transferred style.
- **Web Interface:** Upload content and style images through a Flask application.
- **GPU Support:** Automatically uses CUDA when available and otherwise falls back to the CPU.
- **Custom Training Pipeline:** Train the decoder using content and style image datasets.

## 🧠 Model Architecture

The application follows an AdaIN-based neural style transfer pipeline.

1. **Content image:** Supplies the structure and objects to preserve.
2. **Style image:** Supplies the colors, textures, and artistic patterns.
3. **VGG-19 encoder:** Extracts feature representations from both images.
4. **Adaptive Instance Normalization:** Transfers style statistics to the content features.
5. **Style blending:** Uses an adjustable parameter, `alpha`, to control the strength of the transformation.
6. **Decoder:** Converts the transformed features into the final stylized image.

### Style blending

The implementation blends the transformed features with the original content features:

`target = alpha * stylized_features + (1 - alpha) * content_features`

- `alpha = 0`: Preserve the original content features.
- `alpha = 1`: Apply the full AdaIN transformation.
- Values between 0 and 1: Balance content preservation and transferred style.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning and model inference |
| Torchvision | Image preprocessing and transformations |
| VGG-19 | Feature extraction |
| AdaIN | Feature-based style transfer |
| Flask | Web application backend |
| Bootstrap | Web interface styling |
| Pillow | Image loading and saving |
| uv | Python dependency management |

## 📁 Project Structure

```text
NST/
├── app.py                 # Flask web application
├── main.py                # Entry-point placeholder
├── train.py               # Decoder training pipeline
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency versions
├── decoder_final.pth      # Trained decoder checkpoint
├── vgg_normalised.pth     # VGG encoder weights
├── content_data/          # Content training images
├── style_data/            # Style training images
├── experiment/            # Training configurations and outputs
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── uploads/           # Uploaded and generated images
└── utils/
    ├── models.py          # VGG encoder and decoder
    └── utils.py           # AdaIN and image utilities
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or a compatible environment
- Git
- [uv](https://docs.astral.sh/uv/)
- Sufficient RAM for PyTorch inference; a compatible CUDA GPU is optional

### 1. Clone the repository

```bash
git clone https://github.com/7430souvik/NST.git
cd NST
```

### 2. Install dependencies

Install `uv` if it is not already available:

```bash
pip install uv
```

Sync the project dependencies:

```bash
uv sync
```

### 3. Download the model checkpoints

The application requires these files in the project root:

- `vgg_normalised.pth`
- `decoder_final.pth`

If the repository uses Git LFS for model weights, install Git LFS and retrieve the actual files:

```bash
git lfs install
git lfs pull
```

If the checkpoints are not tracked in the repository, obtain them from the project's designated model download location before starting the application.

### 4. Run the application

```bash
uv run python app.py
```

Open the local URL displayed by Flask, typically:

```text
http://127.0.0.1:10000
```

### 5. Generate a stylized image

1. Upload a content image.
2. Upload a style reference image.
3. Adjust the style intensity.
4. Click **Transfer Style**.
5. View the generated result.

## 🏋️ Training the Decoder

The project includes a custom training pipeline for learning decoder weights.

Prepare content and style image datasets, then run:

```bash
uv run python train.py \
  --content_dir content_data \
  --style_dir style_data \
  --experiment experiment1 \
  --epochs 10 \
  --batch_size 4
```

Training options include:

- `--content_size`: Content image size during preprocessing.
- `--style_size`: Style image size during preprocessing.
- `--final_size`: Final image size.
- `--lr`: Learning rate.
- `--content_weight`: Weight assigned to content loss.
- `--style_weight`: Weight assigned to style loss.
- `--resume`: Resume training from saved checkpoints.

See `train.py` for the complete list of supported arguments.

**Note:** The example uses 10 epochs; adjust the training configuration to your dataset, hardware, and desired results. The training script saves checkpoints according to its configured save interval.

## 📸 Results

Add screenshots of the application and sample transformations here.

Recommended examples:

- Original content image
- Style reference image
- Generated stylized image
- Comparisons using different style-intensity values

<!-- Add committed screenshots below.
![Application Interface](docs/images/app-preview.png)
![Style Transfer Result](docs/images/style-transfer-result.png)
-->

## 🌐 Deployment

The Flask application can be deployed on a compatible Python hosting platform or container environment.

Before deployment:

- Configure the server to bind to `0.0.0.0` and use the platform-provided `PORT`.
- Ensure both model checkpoint files are available.
- Configure persistent or temporary storage for uploaded images as appropriate.
- Keep large model artifacts in Git LFS or a dedicated model-storage service.
- Test inference on the target platform, particularly if GPU acceleration is expected.

**Live demo:** Add your deployed application URL here when available.

## 🔮 Future Improvements

- Add a gallery of example styles.
- Support batch image processing.
- Add image download and result comparison features.
- Improve upload validation and error handling.
- Optimize inference latency and memory consumption.
- Add automated tests for preprocessing and inference.
- Provide a Docker-based deployment workflow.

## 📚 References

- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 👨‍💻 Author

**Souvik Chatterjee**

- GitHub: [@7430souvik](https://github.com/7430souvik)
- Repository: [Neural Style Transfer](https://github.com/7430souvik/NST)


