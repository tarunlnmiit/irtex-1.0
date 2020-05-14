# IRTEX-1.0

## This is the working Scientific project about an Image Retrieval Engine

> The project currently consists of 4 branches:
  1. master - production code ready
  2. develop - development merge of all feature branches
  3. server - feature branch for web server in Django
  4. retrieval-engine - feature branch for image retrieval engine using ML/DL/IR approaches
  
> ### There should not be any incomplete commits in master branch. All developmental code needs to be in develop. Only after complete testing these commits must be merged in the master branch.

>> ### All server changes must be committed to first server branch
>> ### All retrieval engine changes must be committed to first retreival-engine branch

>> ### In order to test the components together develop should be merged into current branch first. When the feature on the current branch is complete then only it needs to be merged into develop. This applies to both feature branches. When all individual branches are working properly then they need to be merged into develop and if all works well then that paricular code will be merged into master branch. 

>> ## Please adhere to the above mentioned protocol

> The application can be initiated in the following manner but it has some prerequisites.

  >> Python (**Use Python 3**) virtualenv must be installed in the system. More details on this link about installation and usage -> https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
  
  * Clone the repository in the virtualenv directory created. (For example the virtualenv directory name is **venv**) 
      
      `cd venv`
      
      **On Linux and MacOS**
      
      `source bin/activate`
      
      **On Windows**
      
      `\env\Scripts\activate.bat`
      
      `git clone https://github.com/tarunlnmiit/irtex-1.0.git`
  * Change into server directory named **irtex** and run the following commands
      ```
      cd irtex
      pip install -r requirements.txt
      ```
      
  * Finally the server can be started using the following command
      `python3 manage.py runserver`
      The server will be default start to run at http://127.0.0.1:8000
  
