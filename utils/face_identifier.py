
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from deepface.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
import represent


def find(img_path, db_path, model_name ='VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True):

	"""
	This function applies verification several times and find an identity in a database
	Parameters:
		img_path: exact image path, numpy array or based64 encoded image. If you are going to find several identities, then you should pass img_path as array instead of calling find function in a for loop. e.g. img_path = ["img1.jpg", "img2.jpg"]
		db_path (string): You should store some .jpg files in a folder and pass the exact folder path to this.
		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib or Ensemble
		distance_metric (string): cosine, euclidean, euclidean_l2
		model: built deepface model. A face recognition models are built in every call of find function. You can pass pre-built models to speed the function up.
			model = DeepFace.build_model('VGG-Face')
		enforce_detection (boolean): The function throws exception if a face could not be detected. Set this to True if you don't want to get exception. This might be convenient for low resolution images.
		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
		prog_bar (boolean): enable/disable a progress bar
	Returns:
		This function returns pandas data frame. If a list of images is passed to img_path, then it will return list of pandas data frame.
	"""

	tic = time.time()

	img_paths, bulkProcess = functions.initialize_input(img_path)

	#-------------------------------

	if os.path.isdir(db_path) == True:

		if model == None:

			if model_name == 'Ensemble':
				print("Ensemble learning enabled")
				models = Boosting.loadModel()

			else: #model is not ensemble
				model = build_model(model_name)
				models = {}
				models[model_name] = model

		else: #model != None
			print("Already built model is passed")

			if model_name == 'Ensemble':
				Boosting.validate_model(model)
				models = model.copy()
			else:
				models = {}
				models[model_name] = model

		#---------------------------------------

		if model_name == 'Ensemble':
			model_names = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']
			metric_names = ['cosine', 'euclidean', 'euclidean_l2']
		elif model_name != 'Ensemble':
			model_names = []; metric_names = []
			model_names.append(model_name)
			metric_names.append(distance_metric)

		#---------------------------------------

		file_name = "representations_%s.pkl" % (model_name)
		file_name = file_name.replace("-", "_").lower()

		if path.exists(db_path+"/"+file_name):

			print("WARNING: Representations for images in ",db_path," folder were previously stored in ", file_name, ". If you added new instances after this file creation, then please delete this file and call find function again. It will create it again.")

			f = open(db_path+'/'+file_name, 'rb')
			representations = pickle.load(f)

			print("There are ", len(representations)," representations found in ",file_name)

		else: #create representation.pkl from scratch
			employees = []

			for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
				for file in f:
					if ('.jpg' in file.lower()) or ('.png' in file.lower()):
						exact_path = r + "/" + file
						employees.append(exact_path)

			if len(employees) == 0:
				raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")

			#------------------------
			#find representations for db images

			representations = []

			pbar = tqdm(range(0,len(employees)), desc='Finding representations', disable = prog_bar)

			#for employee in employees:
			for index in pbar:
				employee = employees[index]

				instance = []
				instance.append(employee)

				for j in model_names:
					custom_model = models[j]

					representation = represent(img_path = employee
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align)

					instance.append(representation)

				#-------------------------------

				representations.append(instance)

			f = open(db_path+'/'+file_name, "wb")
			pickle.dump(representations, f)
			f.close()

			print("Representations stored in ",db_path,"/",file_name," file. Please delete this file when you add new identities in your database.")

		#----------------------------
		#now, we got representations for facial database

		if model_name != 'Ensemble':
			df = pd.DataFrame(representations, columns = ["identity", "%s_representation" % (model_name)])
		else: #ensemble learning

			columns = ['identity']
			[columns.append('%s_representation' % i) for i in model_names]

			df = pd.DataFrame(representations, columns = columns)

		df_base = df.copy() #df will be filtered in each img. we will restore it for the next item.

		resp_obj = []

		global_pbar = tqdm(range(0, len(img_paths)), desc='Analyzing', disable = prog_bar)
		for j in global_pbar:
			img_path = img_paths[j]

			#find representation for passed image

			for j in model_names:
				custom_model = models[j]

				target_representation = represent(img_path = img_path
					, model_name = model_name, model = custom_model
					, enforce_detection = enforce_detection, detector_backend = detector_backend
					, align = align)

				for k in metric_names:
					distances = []
					for index, instance in df.iterrows():
						source_representation = instance["%s_representation" % (j)]

						if k == 'cosine':
							distance = dst.findCosineDistance(source_representation, target_representation)
						elif k == 'euclidean':
							distance = dst.findEuclideanDistance(source_representation, target_representation)
						elif k == 'euclidean_l2':
							distance = dst.findEuclideanDistance(dst.l2_normalize(source_representation), dst.l2_normalize(target_representation))

						distances.append(distance)

					#---------------------------

					if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
						continue
					else:
						df["%s_%s" % (j, k)] = distances

						if model_name != 'Ensemble':
							threshold = dst.findThreshold(j, k)
							df = df.drop(columns = ["%s_representation" % (j)])
							df = df[df["%s_%s" % (j, k)] <= threshold]

							df = df.sort_values(by = ["%s_%s" % (j, k)], ascending=True).reset_index(drop=True)

							resp_obj.append(df)
							df = df_base.copy() #restore df for the next iteration

			#----------------------------------

			if model_name == 'Ensemble':

				feature_names = []
				for j in model_names:
					for k in metric_names:
						if model_name == 'Ensemble' and j == 'OpenFace' and k == 'euclidean':
							continue
						else:
							feature = '%s_%s' % (j, k)
							feature_names.append(feature)

				#print(df.head())

				x = df[feature_names].values

				#--------------------------------------

				boosted_tree = Boosting.build_gbm()

				y = boosted_tree.predict(x)

				verified_labels = []; scores = []
				for i in y:
					verified = np.argmax(i) == 1
					score = i[np.argmax(i)]

					verified_labels.append(verified)
					scores.append(score)

				df['verified'] = verified_labels
				df['score'] = scores

				df = df[df.verified == True]
				#df = df[df.score > 0.99] #confidence score
				df = df.sort_values(by = ["score"], ascending=False).reset_index(drop=True)
				df = df[['identity', 'verified', 'score']]

				resp_obj.append(df)
				df = df_base.copy() #restore df for the next iteration

			#----------------------------------

		toc = time.time()

		print("find function lasts ",toc-tic," seconds")

		if len(resp_obj) == 1:
			return resp_obj[0]

		return resp_obj

	else:
		raise ValueError("Passed db_path does not exist!")

	return None