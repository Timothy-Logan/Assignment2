Timothy Logan Assignment #2 - Systems design
Description: This project is a FastAPI application thats intended to be used for whether forcasting. It allows users to upload historical weather data for a specific area, both in text form or just uploading a file to the program, and have future forcasts based of this information. The data and future predictions can also then be visualized into graphs for the user to see. This API could be used both for meteorologist as a tool to relay weather information, whether that be through an app or a website using this API.
![image](https://github.com/user-attachments/assets/cd4f302e-9259-414f-b290-c73e300c5138)

This is an example of a graph created on the FastAPI local host. 

Now to go more into detail about the FastAPI/docs endpoint, it has four different items on the endpoint list. These items include
- /read root (to check that its running on the correct port, as I was having issues with this during the assignment)
- /forecast (used to take each of the uploaded files and then do forecasting on said information, an example output of this is {
  "lagged_values": [15.2, 16.8, 14.1],
  "location": "Ajax, Ontario"
})
- /forecast/plot (which as you would expect plots the data that its given, the graph above comes from that endpoint)
- /upload (used for the uploading of the files of data)

Below is an image of the /docs
![image](https://github.com/user-attachments/assets/d332ce90-5e21-487a-ae68-2a144119c196)

Now we'll talk about the steps of deployment, and how you could take this code and deploy it on your own system. The following are the things to put into a bash terminal, this is assuming you have docker and all necessary applications installed and running on your system.

cd "path/to/your/project" - ensure you are using the directory with the files

docker build -t fastapi-app .  - create the docker image

docker run -e PORT=8080 -p 8080:8080 fastapi-app  - run the container locally, the url for the /docs of this would be http://localhost:8080/docs

For the Google cloud deployment I used Google artifact registry, as gcr was not operating for me poperly at the time of deploying this, although I had the same issues with GAR and it was just an issue with my code, oh well.

gcloud services enable artifactregistry.googleapis.com  - enable GAR

gcloud artifacts repositories create fastapi-repo \
  --repository-format=docker \
  --location=us-central1     -  create the repository

docker tag fastapi-app us-central1-docker.pkg.dev/trainmodelassigntwo/fastapi-repo/fastapi-app:latest  - tag the docker image

docker push us-central1-docker.pkg.dev/trainmodelassigntwo/fastapi-repo/fastapi-app:latest  - push the image to artifact registry

gcloud run deploy fastapi-app \
  --image us-central1-docker.pkg.dev/trainmodelassigntwo/fastapi-repo/fastapi-app:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated    - deploy the app

  After this is run it will give you a URL for the application, for me this url was https://fastapi-app-30747264556.us-central1.run.app
  ![image](https://github.com/user-attachments/assets/3db8a300-440f-469d-9be9-65320af579e7)

  This an an image of the status page of my API on google cloud services to prove it is up and running.


