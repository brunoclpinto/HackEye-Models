# HackEye-Models
YOLO training for the HackEye app.

## Setup
I'm developing this on a mac. Not using Jupyter notebooks for now — setup and training will run in Docker.

Docker setup instructions TBD.

## Datasets
### Bus
#### OpenImageV7
Don't forget to disable all Options.

https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=valtest&c=%2Fm%2F01bjv
https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F01bjv

##### Get Bus dataset
Loaded the entire page, navigated till the end and made sure all images got loaded correctly along the way.
On chrome, inspected first image, navigated to a top level div that contains all the dataset and copy pasted into `Datasets/openImagesV7-Bus.md`. Did this for both URIs.
Now lets process it so it matches YOLO requirements, follow along 


#### Cityscapes
Sounds like an amazing dataset, gonna try and gain access to it https://www.cityscapes-dataset.com
