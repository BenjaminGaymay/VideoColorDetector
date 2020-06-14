#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from sys import argv
from os import mkdir
from glob import glob
from os.path import exists
from datetime import datetime
from statistics import median_high, stdev, mean


class LightningDetector():
	def __init__(self, tolerance=20, out_directory='./out', show=True, debug=False, threshold=127):
		self.__fd = open('LOG.txt', 'a')
		self.__log = True
		self.tolerance = tolerance
		self.threshold = threshold
		self.out_directory = out_directory.rstrip('/')
		self.show = True
		self.debug = debug

		self.__debugFd = open('DEBUG.txt', 'w') if debug else None

		cv2.namedWindow('Original')
		cv2.namedWindow('Gray')
		cv2.namedWindow('Black And White')
		cv2.moveWindow('Original', 0, 0)
		cv2.moveWindow('Gray', 960, 540)
		cv2.moveWindow('Black And White', 960, 0)

	def __del__(self):
		self.__fd.close()
		if self.__debugFd:
			self.__debugFd.close()

		cv2.destroyAllWindows()

	def LOG(self, message):
		if self.__log:
			print(message, end='')
			self.__fd.write(message)

	def DEBUG(self, message, console=True):
		if self.debug and self.__debugFd:
			print('DEBUG:', message, end='') if console else None
			self.__debugFd.write(message)

	def FLUSH(self):
		self.__fd.flush()

	def get_date(self):
		return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

	def set_tolerance(self, tolerance):
		self.tolerance = tolerance

	def set_show(self, show):
		if show:
			cv2.namedWindow('Original')
			cv2.namedWindow('Gray')
			cv2.namedWindow('Black And White')
			cv2.moveWindow('Original', 0, 0)
			cv2.moveWindow('Gray', 0, 540)
			cv2.moveWindow('Black And White', 960, 0)
		else:
			cv2.destroyAllWindows()

		self.show = show

	def analyse_frame(self, frame):
		height, width, channels = frame.shape

		if width > 960 and height > 540:
			frame = cv2.resize(frame, (width // 4, height // 4))

		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		(thresh, black_and_white_frame) = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY)

		white_pixels_value = cv2.countNonZero(black_and_white_frame)
		if not self.whites_pixel_values:
			if white_pixels_value == 0:
				return True, 0, 0
			self.whites_pixel_values = [white_pixels_value]
			self.LOG(f'First frame White value: {self.whites_pixel_values[0]}\n')

		if self.show:
			self.show_frames(frame, gray_frame, black_and_white_frame)

		variation = (white_pixels_value - mean(sorted(self.whites_pixel_values)[-500:])) / white_pixels_value * 100

		if variation < self.tolerance:
			self.whites_pixel_values.append(white_pixels_value)
			return False, white_pixels_value, variation

		return True, white_pixels_value, variation

	def read_video(self, capture, last_frame):
		frame_counter = 0
		self.stroke_frames = []
		self.whites_pixel_values = []

		image_name = self.video_directory.split('/')[-1]

		errors = 0
		while (capture.isOpened()):
			ret, original_frame = capture.read()
			try:
				frame_counter += 1
				frame = original_frame

				if (frame_counter / last_frame * 100) % 10 == 0:
					print(f'{frame_counter / last_frame * 100}%')

				if frame_counter == 1:
					height, width, channels = frame.shape
					self.LOG(f'Width: {width}\nHeight: {height}\n\n')

				is_stroke, white_pixels_value, variation = self.analyse_frame(frame)

				if is_stroke:
					self.stroke_frames.append(frame_counter)
					self.LOG(
					    f'\t- Frame:\t{frame_counter} {frame_counter - errors}\t\t-\t{white_pixels_value}\t-\t{round(variation)}%\n'
					)

					if not self.debug:
						if not exists(f'{self.video_directory}'):
							mkdir(f'{self.video_directory}')

						cv2.imwrite(f"{self.video_directory}/{image_name}-{frame_counter}.jpg", original_frame)

				if cv2.waitKey(25) & 0xFF == ord('q') or frame_counter == last_frame:
					break

			except Exception as error:
				if frame_counter == last_frame:
					break
				errors += 1

		self.DEBUG(f'median high: {median_high(self.whites_pixel_values)}\n')
		self.DEBUG(f'len: {len(self.whites_pixel_values)}\n')
		self.DEBUG(f'\nwhite values: {self.whites_pixel_values}\n')
		self.DEBUG(f'\nsorted: {sorted(self.whites_pixel_values)}\n\n')

	def show_frames(self, original, gray, black_and_white):
		cv2.imshow('Original', original)
		cv2.imshow('Gray', gray)
		cv2.imshow('Black And White', black_and_white)

	def count_frame_amount(self, capture):
		capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
		last_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)

		capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
		# capture.set(cv2.CAP_PROP_POS_FRAMES, 2500)

		return last_frame

	def read_frame(self, video_path, frame_index):
		capture = cv2.VideoCapture(video_path)

		while True:
			try:
				capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

				print(f'Reading frame {frame_index}', end='\r')

				ret, frame = capture.read()
				height, width, channels = frame.shape
				if width > 960 and height > 540:
					frame = cv2.resize(frame, (width // 4, height // 4))

				gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				(thresh, black_and_white_frame) = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY)

				self.show_frames(frame, gray_frame, black_and_white_frame)

			except Exception as e:
				print(e)
				pass

			k = cv2.waitKey()
			if k == 27 or k == 113:
				break
			elif k == 81:
				frame_index -= 1
			elif k == 83:
				frame_index += 1
			elif k == 85:
				frame_index += 120
			elif k == 86:
				frame_index -= 120

	def analyse_video(self, video_path):
		video_name = video_path.rstrip('/').split('/')[-1]
		self.video_directory = f'{self.out_directory}/{video_name.split(".")[0]}'

		capture = cv2.VideoCapture(video_path)

		self.LOG(f'[+] {self.get_date()} Opening {video_path}..')
		self.DEBUG(f'[+] {self.get_date()} Opening {video_path}\n\n', False)
		if (capture.isOpened() == False):
			self.LOG(' failed.\n\n')
			self.FLUSH()
			return False

		self.LOG(' succeed.\n\n')

		last_frame = self.count_frame_amount(capture)
		fps = capture.get(cv2.CAP_PROP_FPS)

		self.LOG(f'Frame number: {last_frame}\n')
		self.LOG(f'FPS: {fps}\n')
		self.FLUSH()

		self.first_frame = None
		self.read_video(capture, last_frame)

		stroke_time = set([])
		for frame_index in self.stroke_frames:
			s = round(frame_index / fps)
			stroke_time.add('{:02}:{:02}:{:02}'.format(s // 3600, s % 3600 // 60, s % 60))

		self.LOG(f'\nStrokes:\n')
		for time in sorted(stroke_time):
			self.LOG(f'\t- {time}\n')

		capture.release()
		self.LOG(f'\n[-] {self.get_date()} Closing {video_path}\n\n')
		self.DEBUG(f'[-] {self.get_date()} Closing {video_path}\n\n', False)

		return True

	def image_reference(self, image_path):
		self.__log = False
		image = cv2.imread(image_path)
		self.whites_pixel_values = []
		self.analyse_frame(image)
		cv2.waitKey(0)
		self.__log = True

	def analyse_images(self, images):
		for image_path in images:
			image_name = image_path.rstrip('/').split('/')[-1]
			self.video_directory = f'{self.out_directory}/{image_name.split(".")[0]}'

			image = cv2.imread(image_path)

			self.LOG(f'[+] {self.get_date()} Opening {image_path}..')
			self.DEBUG(f'[+] {self.get_date()} Opening {image_path}\n\n', False)

			self.LOG(' succeed.\n\n')

			is_stroke, white_pixels_value = self.analyse_frame(image)

			height, width, channels = image.shape
			self.LOG(f'Width: {width}\nHeight: {height}\n\n')
			self.LOG(f'Strokes: {is_stroke}\n')
			self.LOG(f'White value: {white_pixels_value}\n')
			self.LOG(f'White reference value: {self.white_pixels_reference}\n')

			self.LOG(f'\n[-] {self.get_date()} Closing {image_path}\n\n')
			self.DEBUG(f'[-] {self.get_date()} Closing {image_path}\n\n', False)

		return True


if __name__ == '__main__':

	def usage(exit_value):
		print(f"""USAGE:

{argv[0]} OPTIONS [files] [--r=image_path] [--frame=index]

	--a		video and images analysis
	--d		DEBUG mode
	--e		blacklist following files
	--f		fast mode (without rendering)
	--frame	render specific frame for a video
	--i		images analysis
	--r		set reference for image mode (needed for --i and --a)
	--v		videos analysis""")
		exit(exit_value)

	def globCaseLess(directory, extensions):
		files = []
		for ext in extensions:
			files += glob(f'{directory}/*{ext.lower()}') + glob(f'{directory}/*{ext.upper()}')
		return files

	detector = LightningDetector(tolerance=50, threshold=35, debug='--d' in argv)

	VIDEOS = []
	IMAGES = []
	REFERENCE = None
	FRAME = None
	try:
		for arg in argv[1:]:
			if arg.startswith('--r='):
				REFERENCE = arg.split('--r=')[1]
			elif arg.startswith('--frame='):
				FRAME = int(arg.split('--frame=')[1])
			elif arg.startswith('--'):
				continue
			elif arg.endswith('.mp4') or arg.endswith('.MP4'):
				VIDEOS.append(arg)
			elif arg.lower().endswith('.mp4'):
				VIDEOS.append(arg)
			elif arg.lower().endswith('.jpg') or arg.lower().endswith('.png') or arg.lower().endswith('.jpeg'):
				IMAGES.append(arg)
	except:
		usage(1)

	if '--f' in argv:
		detector.set_show(False)

	if '--v' in argv or '--a' in argv:

		VIDEOS = [file for file in globCaseLess('./videos', ['.mp4'])
		          if file not in VIDEOS] if '--e' in argv or not len(VIDEOS) else VIDEOS

		for video in VIDEOS:
			detector.analyse_video(video)

	if ('--i' in argv or '--a' in argv) and REFERENCE:
		IMAGES = [file for file in globCaseLess('./images', ['.jpg', '.png', '.jpeg'])
		          if file not in IMAGES] if '--e' in argv or not len(IMAGES) else IMAGES

		detector.image_reference(REFERENCE)
		# detector.analyse_images(IMAGES)

	if FRAME != None:
		detector.read_frame(VIDEOS[0], FRAME)

	if '--i' not in argv and '--a' not in argv and '--v' not in argv and FRAME == None or (
	    ('--i' in argv or '--a' in argv) and not REFERENCE):
		usage(1)