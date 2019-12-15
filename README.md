# SignatureRecognition
A fully developed forgery detection system using neural network in python built with django framework.  
## INSTALL DJANGO:    
goto the command line and execute the following commands  
      pip install django  
## MAKE MIGRATIONS:   
Navigate to the file named forgery which contains the manage.py file  
      python manage.py makemigrations signforgery
## MIGRATE    
      python manage.py migrate  
## INSERT SIGNATURE IMAGES(TRAINING SET) RELATED TO FORGERY DETECTION INTO THE DATABASE  
      SHELL  
          python manage.py shell  
      EXECUTE IN SHELL  
          from signforgery.models import Image   
          i=Image(name="personname",classified="genuine/forged",imagefile="filename")  
          i.save()  
## RUN SERVER    
      python manage.py runserver  
## WEBSITE  
To check for forgery in a signature.  
      http://127.0.0.1:8000/home/  
      
