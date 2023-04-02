# Face Recognition

## __Introduction:__  

   In the Project, we are about to perform face recognition using **openCV LBPHFaceRecognizer** python.


## __Project Description__

Root directory of this project contains:
>
> - **1 sub-folders `src`(Contains our necessary scripts)**
> - **README.md file (Contains instructions to run the project)**

Details about the sub-folder and files:
 >
 > - **cascades(folder):**  Contains all the data of `xml` file by `openCV`  
 > - **face_detection(folder):** Contains all the `face detection` modules.
 > - **images(folder):** Contains all the training images to train the model.
 > - **models(folder):** All the pretrained model that are used in this project.
 > - **face_detectors.py**: `Face detection` module to apply `face detection`.
 > - **face_register.py**: Module to register new faces to train with.
  > - **faces_train.py**: Train the algorithm with the new faces.
   > - **faces_recognition.py**: `Module` to `recognize`  faces from `webcam video`.
 > - **main.py**: Our `main` module to do all the operations.
 > - **requirements.txt**: Contains all the dependencies of the project.

>
### __Instructions:__

> First clone the repo using the following command:  
> > `https://github.com/Apucs/face_recognition_LBPH.git`  
>
> Go to the `src` folder:  
> > `cd ./src`
> 
> Before starting we need to satisfy all the dependencies. For that reason need to execute the following command. (All the commands need to be executed from the root folder)
>
> - __Install the dependencies:__  
>> `pip install -r requirements.txt`  

> To `register` the new face, run the following command:
> 
> >`python main.py --action reg`  
>
> To `train` with the new faces, run the following command:
>> `python main.py --action train`  
> 
> To  start the `facial recogition` module, run the following command:
> >  - `python main.py --action recog`  

