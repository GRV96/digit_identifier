import cv2
from pathlib import Path
from sys import argv

from recognition_confidence import\
	RecognitionConfidence


_DOT = "."
_UNDERSCORE = "_"

_PATTERN_DATASET_IMG = "*_*.jpg"


def get_digit_from_file_name(img_file_name):
	dot_index = img_file_name.index(_DOT)
	img_file_name = img_file_name[:dot_index]
	split_name = img_file_name.split(_UNDERSCORE)
	return int(split_name[0])


def create_database(image_dir, loading_flag):
	database = dict()
	for i in range(10): # Digits from 0 to 9
		database[i] = list()

	for img_file in image_dir.glob(_PATTERN_DATASET_IMG):
		digit =  get_digit_from_file_name(img_file.name)
		digit_image = cv2.imread(img_file, loading_flag)
		database[digit].append(digit_image)

	return database


def print_dict(a_dict):
	output = ""

	for k, v in a_dict.items():
		output += f"{k}: {v}\n"

	print(output)


def recognize_digit(digit_image, database):
	confidence = RecognitionConfidence()

	for digit, db_images in database.items():
		for db_image in db_images:
			size0, size1 = db_image.shape
			for i0 in range(size0):
				for i1 in range(size1):
					if digit_image[i0, i1] == db_image[i0, i1]:
						confidence.apply_delta_score(digit, 2)
					else:
						confidence.apply_delta_score(digit, -1)

	return confidence.calculate_confidence()


digit_path = argv[1]

db = create_database(Path.cwd()/"dataset", cv2.IMREAD_GRAYSCALE)

image = cv2.imread(digit_path, cv2.IMREAD_GRAYSCALE)
digit, confidence = recognize_digit(image, db)

print(f"Digit: {digit}\nConfidence: {confidence}")
