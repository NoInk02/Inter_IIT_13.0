
## Adobe- Image Classification and Artifact Detection

## INPUT FOLDER NOT UPLOADED AS TOO BIG OF A FOLDER

This repository contains a collection of machine learning models and associated scripts for the artifact detection task, using various architectures. The main directory contains the necessary files for running the final model, while additional models (MobileNet, ResNet50, VGG16, fine_tuned_vilt_models and ViT) are stored in separate folders for experimentation and comparison.



## Directory Structure

The submitted directory looks like this:

```bash
final/
│
├── main/
│   ├── code/
│   │   ├── Main_(FINALE).ipynb       # Main file to be run for output
│   │   ├── output/                   # Folder containing output files
│   │   │   ├── team_83_task1.json    # Predictions on the test data
│   │   │   └── team_83_task2.json    # Explanations of detected artifacts
|   |   |
│   │   ├── input/                    # Folder containing all the required inputs for testing
|   |   |
│   │   ├── Task_1/                   # Files related to Task 1
│   │   │   ├── adversarial/          # Adversarial training-related files
│   │   │   └── classifier_training_testing/  # Training and testing Hybrid Model 
│   |   |
│   │   └── Task_2/                   # Files related to Task 2
│   │   │   ├── gradcam_pos/          # Files related to Grad-CAM for positive classes
│   │   │   └── Single_Image_Inference_MiniCPM_V_1,2_3_43B      # File containing code for artefact detection and explanation
|   |   |
│   └── models/                       # Folder containing all of our saved models required for the Main_(FINALE) file
│
└── other_models/
    ├── mobilenet/
    │   ├── code/                     # Code specific to MobileNet
    │   └── results/                  # Results specific to MobileNet
    ├── resnet50/
    │   ├── code/                     # Code specific to ResNet50
    │   └── results/                  # Results specific to ResNet50
    ├── vgg16/
    │   ├── code/                     # Code specific to VGG16
    │   └── results/                  # Results specific to VGG16
    |── vit/
    |    ├── code/                    # Code specific to Vision Transformer
    |    └── results/                 # Results specific to Vision Transformer
    | 
    |── adversarial.h5                # Our adversarial trained hybrid model 
    └──fine_tuned_vilt_models         # Models fine tuned on our custom dataset 
         |
         |── ai-artifacts.txt         # File containing the given set of artifacts
         ├── dataset.csv/             # Our custom dataset created by us
         ├── image_data/              # ai-generated images with the artifacts                                 
         ├── fine_tuned_40_12/
         │   ├── fine_tuned_40_12.ipynb  # code of our fine tuning of the model with 1/2 augmentation 
         │   ├── implement_40_12.ipynb   # loading of our model predicting the artifacts for a single image
         |   └── fine_tuned_model_40_12/ # directory contating all the required files for the model                
         ├── fine_tuned_40_14/                     
         │   ├── fine_tuned_40_14.ipynb  # code of our fine tuning of the model with 1/4 augmentation 
         │   ├── implement_40_14.ipynb   # loading of our model predicting the artifacts for a single image
         |   └── fine_tuned_model_40_14/ # directory contating all the required files for the model                
         ├── fine_tuned_40_13/ 
         │   ├── fine_tuned_40_13.ipynb  # code of our fine tuning of the model with 1/3 augmentation 
         │   ├── implement_40_13.ipynb   # loading of our model predicting the artifacts for a single image
         |   └── fine_tuned_model_40_13/ # directory contating all the required files for the model                
         └── fine_tuned_40_23/
             ├── fine_tuned_40_23.ipynb  # code of our fine tuning of the model with 2/3 augmentation 
             ├── implement_40_23.ipynb   # loading of our model predicting the artifacts for a single image
             └── fine_tuned_model_40_23/ # directory contating all the required files for the model                
                     
                  

```

## Files Description:
- **final/main/code/Main_(FINALE).ipynb:** our main code file that needs to be run to get the results  
- **final/main/code/output/team_83_task1.json:** contains the output for Task_1
- **final/main/code/output/team_83_task2.json:** contains the output for Task_2
- **final/main/code/input/ai-artifacts.txt:** contains the artifacts list given by the organizers
- **final/main/code/input/preturbed_images_32:** contains the image dataset provided by the organizers
- **classifier_training_testing:** contains the file where we have trained and tested our hybrid model
- **final/main/code/Task_1/Adversaraial.ipynb:** contains the code where we have adversarially trained our models
- **final/main/code/Task_2/gradcam_pos.ipynb:** contains the file where we have implemented gradcam for model interpretability 
- **final/main/code/Task_2/Single_Image_Inference_MiniCPM_V_1,2_3_43B.ipynb:** contains the file where artefact detection and explanation can be generated for a single image
- **final/main/EDSR_x4.pb:** model that produces high resolution upscaled images 
- **final/main/hybrid_resnet_wavelet_model_saved.h5:** our hybrid model that is used for classifying images as ai generated or real
- **final/other_models/mobilenet/code.ipynb:** code for our fine tuned mobilenet model for image classification
- **final/other_models/resnet/code.ipynb:** code for our fine tuned resnet model for image classification
- **final/other_models/vgg16/code.ipynb:** code for our fine tuned vgg16 model for image classification
- **final/other_models/vit/code.ipynb:** code for our fine tuned vit model for image classification


## Output Files:
The output of the given test dataset is present in "**/final/main/code/Output/**" as **team_83_task1.json** and **team_83_task2.json** 
## Running the Main File
- **To generate the outputs you just have to run the main file present at final/main/code/Main_(FINALE).ipynb:**
- **Main file can be run easily on collab, but to run locally follow the below step**

To run our main file **Main_(FINALE).ipynb** locally following steps are needed to be followed:
- Create a virtual enviornment with python version 3.10.12
- Make sure your machine has both CuDNN drivers for tensorlfow, also drivers compatible for the torch. This is beacuse on Cloud (Run Pod) for pytorch template this file will show error ( Tensorflow driver's error. ) 
- Locally use VS Code with VIRE (Enviroment)

```bash
    pip install --upgrade pip
    pip install matplotlib
    pip install seaborn
    pip install scikit-learn
    pip install opencv-python
    pip install pillow
    pip install transformers==4.40.0
    pip install torch
    pip install torchvision
    pip install pywavelets
    pip install tqdm
    pip install opencv-python==4.6.0.66
    pip install opencv-contrib-python==4.6.0.66
    pip install gputil
    pip install timm
    pip install peft
    pip install sentencepiece

```
- After importing all the dependencies, following models are to be loaded which are present in **final/main/code/models/**
  - **EDSR_x4.pb**
  - **hybrid_resnet_wavelet_model_saved.h5**
  Give correct path (according to your machine), code for loading the models is provided in **Main_(FINALE)**
- Provide the correct path for:
  - **artifacts.txt path:** final/main/code/input/ai-artifacts.txt
  - **Your respective image folder**

- **To load the finetuned Vilt code is provided on the main file section. Base-vilt is set as default.**

## Other Models

- **Task 1:**
    
    For Task 1 we are using our own HYBRID MODEL but other models can be used, the codes for which are present in **other_model** folder of zip file.
    
    Code for each model is present in code file of each model

    Other models we experimented are present in:
   - **MobileNet:** Located in the **mobilenet/** folder.
    - **ResNet50:** Located in the **resnet50/** folder.
    - **VGG16:** Located in the **vgg16/** folder.
    - **Vision Transformer (ViT):** Located in the **vit/** folder.

- **Task 2:**
    
    - **Artifact Detection**
        
        We tried to fine tune the **vilt-b32-finetuned-vqa** with our own custom dataset, which we created ourselves. But due to small size of the dataset, these models were not able to give a good result.

        We have attached the file that we used to fine tune the model along with the fine tuned model in **other_models/fine_tuned_vilt_models/**

    
    - **Explanation**
     
        For explanation generation we are using **MiniCPM-V-2** which is 3.43B in size but you can also use other models by:

    * For using **MiniCPM 8.1B** use

        ```
        AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True, torch_dtype=torch.bfloat16)
        AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

        ```
    * For using **MiniCPM Llama V 2.5  - 8.54B**

        ```
        AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
        AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
        ```

- **fine_tuned_vilt_model/**
    - This directory contains many models and code file for them:
    - **image_data/:** contains the images used for fine tuning the model
    - **dataset.csv:** dataset generated by us to fine tune the model
    - **fine_tuned_model_40_12/:** it contains the model fine tuned on 40 image dataset with 1/2 augmentation 
    - **fine_tuned_model_40_13/:** it contains the model fine tuned on 40 image dataset with 1/3 augmentation  
    - **fine_tuned_model_40_14/:** it contains the model fine tuned on 40 image dataset with 1/4 augmentation
    - **fine_tuned_model_40_23/:** it contains the model fine tuned on 40 image dataset with 2/3 augmentation
The code for each fine tuned model with the results can be found in that models respective .ipynb file.
