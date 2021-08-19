for prog in "cascade_facelib" "convnet_facelib" "mtcnn_facelib" "facelib_facelib" "detectron_facelib" "mtcnn-detectron_facelib" "cascade_convnet" "convnet_convnet" "mtcnn_convnet" "facelib_convnet" "detectron_convnet" "mtcnn-detectron_convnet"
do
echo -e $prog
for occ in "bio" "ceo" "cook" "eng" "nur" "pol" "pst" "pro" "sd" "td"
do
echo -e $occ
python eval.py ./Results/$prog/$occ.txt ./ground_truth/$occ.txt
done
done



