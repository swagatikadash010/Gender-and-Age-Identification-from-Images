for prog in "cascade_facelib" "convnet_facelib" "mtcnn_facelib" "facelib_facelib" "detectron_facelib" "mtcnn-detectron_facelib" "cascade_convnet" "convnet_convnet" "mtcnn_convnet" "facelib_convnet" "detectron_convnet" "mtcnn-detectron_convnet"
do
echo -e $prog
echo -e "++++++++++++++++"
#mkdir ../Results/$prog
for occ in "cook" "eng" "nur" "pol" "pst" "pro" "sd" "td"
do
echo -e $occ
python $prog.py ../Ranked_Images/$occ/* >../Results/$prog/$occ.txt
done
done



