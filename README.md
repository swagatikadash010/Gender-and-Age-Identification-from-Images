## Gender Identification from Images Using Python

**Author**: Swagatika Dash

Add details here.

### Installation Requirements

Add instructions here. 

### How to run ?

1. Go to the `src` directory. 
2. Run a script like this:

	python mtcnn_convnet.py <path_to_directory>/*.jpg >mtcnn_convnet_output.txt

3. For Amazon's gender detection an example command is this:

	python amazon_rekognition.py <directory_name_in_bucket> >amazon_output.txt
	
4. For evaluation results run this from the parent directory (not `src`):

	python eval.py <ground_truth_file_name>.txt <prediction_file_name>.txt
