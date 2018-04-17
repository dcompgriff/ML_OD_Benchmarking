README:

Detectron caffe2 CPU based network link:
https://github.com/caffe2/models/tree/master/detectron/e2e_faster_rcnn_R-50-C4_1x

Detectron Models We Will Test:
Faster-RCNN Models:
	*R-50-C4 2x
	*R-50-FPN 2x
	*R-101-FPN 2x
	*X-101-64x4d-FPN 2x
	*X-101-32x8d-FPN 2x
Mask-RCNN Models:
	*R-50-C4 2x
	*R-50-FPN 2x
	*R-101-FPN 2x
	*X-101-64x4d-FPN 2x
	*X-101-32x8d-FPN 2x
Retina-Net Models:
	*R-50-FPN 2x
	*R-101-FPN 2x
	*X-101-64x4d-FPN 2x
	*X-101-32x8d-FPN 2x
The "Big-Boy":
	X-152-32x8d-FPN-IN5k




Kinds of image transforms
1) Cropping, affine transforms, rotations, scale while keeping aspect ratio, etc...
2) Image region blackout, blurring, saturation, contrast, 
absolute value change
3) Random, Gaussian, Poisson, Shot noise
4) Lighting variation, saturation, other image editing types of image changes, 
https://en.wikipedia.org/wiki/Image_editing#Slicing_of_images
https://en.wikipedia.org/wiki/Image_noise
5) Distortions https://en.wikipedia.org/wiki/Distortion_(optics)


Caffe2 Installation Details:
1) Download anaconda2, and perform the anaconda based install (and custom install after that).


COCO data and COCOApi Python Tools:
1) Clone the git repository.
2) cd to "PythonAPI", and run "make"
3) Download images here "http://images.cocodataset.org/zips/test2017.zip" (Test image set)
4) Download annotations here "http://images.cocodataset.org/annotations/image_info_test2017.zip" (Test image annotations)



Image Object Detection Output:
[
  {
    img_name: "",
    score:[],
    bbox:[],
    classes:[]
  },
  {
    img_name: "",
    score:[],
    bbox:[],
    classes:[]
  },
  ...
]


# Commands to clean up the docker stuff if the filesystem gets too full.
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker ps -aq)



