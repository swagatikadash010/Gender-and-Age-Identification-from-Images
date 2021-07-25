from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from collections import Counter

def process_file(fin):
	with open(fin) as f:
		data = {}
		for line in f:
			line = line.strip()
			if line!="":
				fn, val = line.split(",")
				data[fn] = val
	return data
	
if __name__ =="__main__":
	import sys
	pred_file = sys.argv[1]
	ref_file = sys.argv[2]
	
	d_pred = process_file(pred_file) 
	d_ref = process_file(ref_file)
	
	dk1 = list(d_pred.keys())
	dk2 = list(d_ref.keys())
	
	dk1.sort()
	dk2.sort()
	
	pred = [d_pred[i] for i in dk1]
	ref = [d_ref[i] for i in dk1]
	
	
	print ("REF Frequency", Counter(ref))
	print ("Pred Frequency", Counter(pred))
	
	print (precision_recall_fscore_support(ref, pred, average='macro'))
