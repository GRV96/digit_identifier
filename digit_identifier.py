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
	for i in range(10): # Digits from 0 to 10
		database[i] = list()

	for img_file in image_dir.glob(_PATTERN_DATASET_IMG):
		digit =  get_digit_from_file_name(img_file.name)
		digit_image = cv2.imread(img_file, loading_flag)
		database[digit].append(digit_image)

	return database


def recognize_digit(digit_image, database):
	confidence = RecognitionConfidence()

	for digit, ref_images in database.items():
		for ref_image in ref_images:
			for column_index in range(len(ref_image)):
				column = ref_image[column_index]
				for row_index in range(len(column)):
					if digit_image[column_index][row_index] == digit_image[column_index][row_index]:
						confidence.apply_delta_points(digit, 2)
					else:
						confidence.apply_delta_points(digit, -1)

	return confidence.calculate_confidence()


digit_path = argv[1]

db = create_database(Path.cwd()/"dataset", cv2.IMREAD_GRAYSCALE)

image = cv2.imread(digit_path, cv2.IMREAD_GRAYSCALE)
digit, confidence = recognize_digit(image, db)

print(f"Digit: {digit}\nConfidence: {confidence}")
