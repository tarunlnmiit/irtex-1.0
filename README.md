# IRTEX-1.0

* The deployed project can be reached through https://irtex-client.herokuapp.com

## This is the working Scientific project about an Image Retrieval Engine

> The project currently consists of 5 branches:
  1. master - production code ready
  2. develop - development merge of all feature branches
  3. server - feature branch for web server in Django
  4. retrieval-engine - feature branch for image retrieval engine using ML/DL/IR approaches
  5. milestone2 - deliverable branch for milestone 2.
  
> The application can be initiated in the following manner but it has some prerequisites.

  >> Python (**Use Python 3**) virtualenv must be installed in the system. More details on this link about installation and usage -> https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
  
  * Clone the repository in the virtualenv directory created and checkout to **milestone2** branch. (For example the virtualenv directory name is **venv**) 
      
      `cd venv`
      
      **On Linux and MacOS**
      
      `source bin/activate`
      
      **On Windows**
      
      `\env\Scripts\activate.bat`
      
      **The git commands**
      
      `git clone https://github.com/tarunlnmiit/irtex-1.0.git`
      
      `git checkout milestone2`
      
      `git pull origin milestone2`
  
  * Run the following command to install the dependencies.

      `pip install -r requirements.txt`
      
  * Finally the server can be started using the following command

      `python3 manage.py runserver`
      
      The server will be started at http://127.0.0.1:8000
  
  >> The code currently consists of Color Layout Descriptor(CLD) generator and Region Based Shape Descriptor(RBSD) generator. The results sent to the frontend are currently ranked on the average similarities computed on CLD and RBSD along with the individual ranked results of CLD and RBSD.
  
   * In order to run the CLD extractor file, the command is as follows:
      First navigate to directory **color_layout_descriptor**.
      There are 3 inputs needed for this file to execute. 
      1. --path - input images path
      2. --output - path for features output
      3. --type - features to be output in csv or a pkl (pickle) file
           
     `python3 CLDescriptor.py --path <dataset path> --output <output path> --type <feature output type csv or pkl>`
     
   * In order to run the CLD extractor file on *toy dataset*, the command is as follows:
      First navigate to directory **color_layout_descriptor**.
      There is 1 input needed for this file to execute. 
      1. --test-image - query image path
     
     `python3 cld_test.py --test-image <query image path>`
     
     If no query image is provided then one image from toy dataset is taken as query.

   * In order to run the RBSD extractor file, the command is as follows:
      First navigate to directory **region_based_descriptor**.
      There are 3 inputs needed for this file to execute. 
      1. --path - input images path
      2. --output - path for features output
      3. --type - features to be output in csv or a pkl (pickle) file
      
     `python3 MomentsExtractor.py --path <dataset path> --output <output path> --type <feature output type csv or pkl>`
     
   * In order to run the RBSD extractor file on *toy dataset*, the command is as follows:
      First navigate to directory **region_based_descriptor**.
      There is 1 input needed for this file to execute. 
      1. --path - query image path
     
     `python3 rbsd_test.py --path <query image path>`
     
     If no query image is provided then one image from toy dataset is taken as query.
     
   *  Since the csv file generated are big in size and we are internally using pickle to read the extracted feature vectors, we have uploaded the pre-generated csv files on Google Drive.
      
      [RBSD feature csv](https://drive.google.com/file/d/1Scxi92KdOyhW_-G1DCyGmOFmcqoFaUa2/view?usp=sharing)
      
      [CLD feature csv](https://drive.google.com/file/d/1Y4SBJpHMyAMGNBTII5TnF7TuTyaLMb8n/view?usp=sharing)

  >> The flow of the whole web application from a user's perspective is given in the repository of frontend. Here is the link for the same. https://github.com/tarunlnmiit/irtex-client
