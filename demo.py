import cv2, time
import numpy as np
import re
import logging
import pycuda.driver as drv
# from dataFrame import *
import pandas as pd
from cap_from_youtube import  cap_from_youtube

from taskConditions import TaskConditions, Logger
from ObjectDetector.yoloDetector import YoloDetector
from ObjectDetector.utils import ObjectModelType,  CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector
from TrafficLaneDetector.ultrafastLaneDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from TrafficLaneDetector.ultrafastLaneDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ultrafastLaneDetector.utils import LaneModelType, OffsetType, CurvatureType
LOGGER = Logger(None, logging.INFO, logging.INFO )

for i in range(141,151):
	video_path = f"TrafficLaneDetector/temp/Case3&3New/CLV{i}.mp4"
	lane_config = {
		"model_path": "./TrafficLaneDetector/models/culane_res34.trt",
		"model_type" : LaneModelType.UFLDV2_CULANE
	}

	object_config = {
		"model_path": './ObjectDetector/models/yolov8l-coco.trt',
		"model_type" : ObjectModelType.YOLOV8,
		"classes_path" : './ObjectDetector/models/coco_label.txt',
		"box_score" : 0.4,
		"box_nms_iou" : 0.45
	}

	# Priority : FCWS > LDWS > LKAS
	class ControlPanel(object):
		CollisionDict = {
							CollisionType.UNKNOWN : (0, 255, 255),
							CollisionType.NORMAL : (0, 255, 0),
							CollisionType.PROMPT : (0, 102, 255),
							CollisionType.WARNING : (0, 0, 255)
						}

		OffsetDict = {
						OffsetType.UNKNOWN : (0, 255, 255),
						OffsetType.RIGHT :  (0, 0, 255),
						OffsetType.LEFT : (0, 0, 255),
						OffsetType.CENTER : (0, 255, 0)
					 }

		CurvatureDict = {
							CurvatureType.UNKNOWN : (0, 255, 255),
							CurvatureType.STRAIGHT : (0, 255, 0),
							CurvatureType.EASY_LEFT : (0, 102, 255),
							CurvatureType.EASY_RIGHT : (0, 102, 255),
							CurvatureType.HARD_LEFT : (0, 0, 255),
							CurvatureType.HARD_RIGHT : (0, 0, 255)
						}

		def __init__(self):
			collision_warning_img = cv2.imread('./assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
			self.collision_warning_img = cv2.resize(collision_warning_img, (100, 100))
			collision_prompt_img = cv2.imread('./assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
			self.collision_prompt_img = cv2.resize(collision_prompt_img, (100, 100))
			collision_normal_img = cv2.imread('./assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
			self.collision_normal_img = cv2.resize(collision_normal_img, (100, 100))
			left_curve_img = cv2.imread('./assets/left_turn.png', cv2.IMREAD_UNCHANGED)
			self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
			right_curve_img = cv2.imread('./assets/right_turn.png', cv2.IMREAD_UNCHANGED)
			self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
			keep_straight_img = cv2.imread('./assets/straight.png', cv2.IMREAD_UNCHANGED)
			self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
			determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
			self.determined_img = cv2.resize(determined_img, (200, 200))
			left_lanes_img = cv2.imread('./assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
			self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
			right_lanes_img = cv2.imread('./assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
			self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))


			# FPS
			self.fps = 0
			self.frame_count = 0
			self.start = time.time()

			self.curve_status = None

		def _updateFPS(self) :
			"""
			Update FPS.

			Args:
				None

			Returns:
				None
			"""
			self.frame_count += 1
			if self.frame_count >= 30:
				self.end = time.time()
				self.fps = self.frame_count / (self.end - self.start)
				self.frame_count = 0
				self.start = time.time()

		def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25) :
			"""
			Display BirdView Panel on image.

			Args:
				main_show: video image.
				min_show: bird view image.
				show_ratio: display scale of bird view image.

			Returns:
				main_show: Draw bird view on frame.
			"""
			W = int(main_show.shape[1]* show_ratio)
			H = int(main_show.shape[0]* show_ratio)

			min_birdview_show = cv2.resize(min_show, (W, H))
			min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 添加边框
			main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]: ] = min_birdview_show
			return main_show

		def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
			"""
			Display Signs Panel on image.

			Args:
				main_show: image.
				offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
				curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

			Returns:
				main_show: Draw sings info on frame.
			"""

			W = 400
			H = 365
			widget = np.copy(main_show[:H, :W])
			widget //= 2
			widget[0:3,:] = [0, 0, 255]  # top
			widget[-3:-1,:] = [0, 0, 255] # bottom
			widget[:,0:3] = [0, 0, 255]  #left
			widget[:,-3:-1] = [0, 0, 255] # right
			main_show[:H, :W] = widget

			if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
				y, x = self.determined_img[:,:,3].nonzero()
				main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
				self.curve_status = None

			elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
				(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
				y, x = self.left_curve_img[:,:,3].nonzero()
				main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
				self.curve_status = "Left"

			elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
				(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
				y, x = self.right_curve_img[:,:,3].nonzero()
				main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
				self.curve_status = "Right"


			if ( offset_type == OffsetType.RIGHT ) :
				y, x = self.left_lanes_img[:,:,2].nonzero()
				main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
			elif ( offset_type == OffsetType.LEFT ) :
				y, x = self.right_lanes_img[:,:,2].nonzero()
				main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
			elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
				y, x = self.keep_straight_img[:,:,3].nonzero()
				main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
				self.curve_status = "Straight"

			self._updateFPS()
			cv2.putText(main_show, "LDWS : " + offset_type.value, (10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.OffsetDict[offset_type], thickness=2)
			cv2.putText(main_show, "LKAS : " + curvature_type.value, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=self.CurvatureDict[curvature_type], thickness=2)
			cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10, widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
			return main_show

		def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25) :
			"""
			Display Collision Panel on image.

			Args:
				main_show: image.
				collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
				obect_infer_time: object detection time -> float.
				lane_infer_time:  lane detection time -> float.

			Returns:
				main_show: Draw collision info on frame.
			"""
			W = int(main_show.shape[1]* show_ratio)
			H = int(main_show.shape[0]* show_ratio)

			widget = np.copy(main_show[H+20:2*H, -W-20:])
			widget //= 2
			widget[0:3,:] = [0, 0, 255]  # top
			widget[-3:-1,:] = [0, 0, 255] # bottom
			widget[:,-3:-1] = [0, 0, 255] #left
			widget[:,0:3] = [0, 0, 255]  # right
			main_show[H+20:2*H, -W-20:] = widget

			if (collision_type == CollisionType.WARNING) :
				y, x = self.collision_warning_img[:,:,3].nonzero()
				main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]
			elif (collision_type == CollisionType.PROMPT) :
				y, x =self.collision_prompt_img[:,:,3].nonzero()
				main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
			elif (collision_type == CollisionType.NORMAL) :
				y, x = self.collision_normal_img[:,:,3].nonzero()
				main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

			cv2.putText(main_show, "FCWS : " + collision_type.value, ( main_show.shape[1]- int(W) + 100 , 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=self.CollisionDict[collision_type], thickness=2)
			cv2.putText(main_show, "object-infer : %.2f s" % obect_infer_time, ( main_show.shape[1]- int(W) + 100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
			cv2.putText(main_show, "lane-infer : %.2f s" % lane_infer_time, ( main_show.shape[1]- int(W) + 100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
			return main_show


	if __name__ == "__main__":

		# Initialize read and save video
		cap = cv2.VideoCapture(video_path)

		width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		vout = cv2.VideoWriter(video_path[:-4]+'_out.mp4', fourcc , 30.0, (width, height))
		cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)

		#==========================================================
		#					Initialize Class
		#==========================================================
		LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
		LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))

		# lane detection model
		LOGGER.info("UfldDetector Model Type : {}".format(lane_config["model_type"].name))
		if ( "UFLDV2" in lane_config["model_type"].name) :
			UltrafastLaneDetectorV2.set_defaults(lane_config)
			laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
		else :
			UltrafastLaneDetector.set_defaults(lane_config)
			laneDetector = UltrafastLaneDetector(logger=LOGGER)
		transformView = PerspectiveTransformation( (width, height) )

		# object detection model
		LOGGER.info("YoloDetector Model Type : {}".format(object_config["model_type"].name))
		YoloDetector.set_defaults(object_config)
		objectDetector = YoloDetector(logger=LOGGER)
		distanceDetector = SingleCamDistanceMeasure()

		# display panel
		displayPanel = ControlPanel()
		analyzeMsg = TaskConditions()




		count = 0
		df = pd.DataFrame(columns=["id", 'offset',
								   "xmin",
								   "ymin",
								   "xmax",
								   "ymax",
								   "label",
								   # "distance",
								   "car_direction",
								   "car_curv",
								   'lane1_detected',
								   "lane2_detected",
								   "lane3_detected",
								   "lane4_detected",
								   'lane1_points',
								   "lane2_points",
								   "lane3_points",
								   "lane4_points"
								   ])
		## add: objectDetctor.object_info[],
		#, "className"



		while cap.isOpened():

			ret, frame = cap.read() # Read frame from the video
			if ret:
				frame_show = frame.copy()

				#========================== Detect Model =========================
				obect_time = time.time()
				objectDetector.DetectFrame(frame)
				obect_infer_time = round(time.time() - obect_time, 2)
				lane_time = time.time()
				laneDetector.DetectFrame(frame)
				lane_infer_time = round(time.time() - lane_time, 4)

				#========================= Analyze Status ========================
				distanceDetector.calcDistance(objectDetector.object_info)
				vehicle_distance = distanceDetector.calcCollisionPoint(laneDetector.draw_area_points)

				analyzeMsg.UpdateCollisionStatus(vehicle_distance, laneDetector.draw_area)


				if (not laneDetector.draw_area or analyzeMsg.CheckStatus()) :
					transformView.updateTransformParams(laneDetector.lanes_points[1], laneDetector.lanes_points[2], analyzeMsg.transform_status)
				birdview_show = transformView.transformToBirdView(frame_show)

				birdview_lanes_points = [transformView.transformToBirdViewPoints(lanes_point) for lanes_point in laneDetector.lanes_points]
				(vehicle_direction, vehicle_curvature) , vehicle_offset = transformView.calcCurveAndOffset(birdview_show, birdview_lanes_points[1], birdview_lanes_points[2])

				analyzeMsg.UpdateOffsetStatus(vehicle_offset)
				analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

				#========================== Draw Results =========================
				transformView.DrawDetectedOnBirdView(birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
				if (LOGGER.clevel == logging.DEBUG) : transformView.DrawTransformFrontalViewArea(frame_show)
				laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
				frame_show = laneDetector.DrawAreaOnFrame(frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
				objectDetector.DrawDetectedOnFrame(frame_show)
				distanceDetector.DrawDetectedOnFrame(frame_show)

				frame_show = displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
				frame_show = displayPanel.DisplaySignsPanel(frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)
				frame_show = displayPanel.DisplayCollisionPanel(frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time )


				classList = [i for i in objectDetector.object_info]
				# print(classList[0][0])
				distanceList = [i for i in distanceDetector.distance_points]
				# print(distanceList[0][2])
				count += 1
				counter = -1
				for i in classList:
					counter += 1
					newRow = {'id': count,
							  "offset": vehicle_offset,
							  "xmin":i[0][0],
							  "ymin":i[0][1],
							  "xmax":i[0][2],
							  "ymax":i[0][3],
							  "label":i[0][4],
							  # "distance": 0,
							  "car_direction": vehicle_direction,
							  'car_curv': vehicle_curvature,
							  'lane1_detected':  laneDetector.lanes_detected[0],
							  "lane2_detected":laneDetector.lanes_detected[1],
							  "lane3_detected":laneDetector.lanes_detected[2],
							  "lane4_detected":laneDetector.lanes_detected[3],
							  'lane1_points': laneDetector.lanes_points[0],
							  "lane2_points": laneDetector.lanes_points[1],
							  "lane3_points": laneDetector.lanes_points[2],
							  "lane4_points": laneDetector.lanes_points[3]
							  }

					df.loc[len(df)] = newRow
					# print(counter)
					# print(distanceDetector.distance_points[counter][2])
					# if counter > len(classList)-2:
					# 	pass
					#
					# else:
					# 	df['distance'].iloc[-1] = distanceDetector.distance_points[counter][2]
				# print(df['distance'])
				#
				# print(df)
				cv2.imshow("ADAS Simulation", frame_show)

			else:
				break

			vout.write(frame_show)

			if cv2.waitKey(1) == ord('q'): # Press key q to stop
				break

		vout.release()
		cap.release()
		cv2.destroyAllWindows()
		# lane Three
		firstPointList = []
		midPointList = []
		lastPointList = []
		for i in df['lane2_points']:
			foundlist = re.findall(r'\(\d{3},\s\d{3}\)', str(i))
			if foundlist == []:
				firstPointList.append(None)
				midPointList.append(None)
				lastPointList.append(None)
			else:
				firstPointList.append(foundlist[0].strip(r'()'))
				midPointList.append(foundlist[int(len(foundlist) / 2)].strip(r"()"))
				lastPointList.append(foundlist[-1].strip(r"()"))
		#     print(foundpat.group().strip(r'()'))
		df['lane_Two_firstPoint'] = firstPointList
		df['lane_Two_midPoint'] = midPointList
		df['lane_Two_lastPoint'] = lastPointList

		firstPointList = []
		midPointList = []
		lastPointList = []
		for i in df['lane3_points']:
			foundlist = re.findall(r'\(\d{3},\s\d{3}\)', str(i))

			if foundlist == []:
				firstPointList.append(None)
				midPointList.append(None)
				lastPointList.append(None)
			else:
				firstPointList.append(foundlist[0].strip(r'()'))
				midPointList.append(foundlist[int(len(foundlist) / 2)].strip(r"()"))
				lastPointList.append(foundlist[-1].strip(r"()"))

		df['lane_Three_firstPoint'] = firstPointList
		df['lane_Three_midPoint'] = midPointList
		df['lane_Three_lastPoint'] = lastPointList
		df = df.drop(df[(df['car_curv'] > 10000)].index)
		df = df.drop(["lane1_detected",
					  "lane2_detected",
					  "lane3_detected",
					  "lane4_detected",
					  "lane1_points",
					  "lane2_points",
					  "lane3_points",
					  "lane4_points"], axis=1)
		df = df.drop(df[(df['car_curv'] > 10000)].index)
		df = df.reset_index()

		# Keep only data with car label
		df = df[df['label'] == 'car']
		# Delete duplicated id
		df = df.drop_duplicates(subset='id', keep='first')
		# Drop null
		df = df.dropna(thresh=df.shape[1] - 2)
		# Get first , middle and last frame in each second
		# Start from bottom to up
		step = 5
		start_index = len(df) - 1
		df = df.iloc[start_index::-step]
		df = df[::-1]
		# Separate X & Y in each point
		columns = ['lane_Two_firstPoint', 'lane_Two_midPoint', 'lane_Two_lastPoint', 'lane_Three_firstPoint',
				   'lane_Three_midPoint', 'lane_Three_lastPoint']
		for column in columns:
			df[['x_' + column, 'y_' + column]] = df[column].str.split(',', expand=True)
			df[['x_' + column, 'y_' + column]] = df[['x_' + column, 'y_' + column]].astype(float)
			df[['x_' + column, 'y_' + column]] = df[['x_' + column, 'y_' + column]].interpolate(method='linear')
			df[['x_' + column, 'y_' + column]] = df[['x_' + column, 'y_' + column]].astype(int)
		# Drop Uneeded columns
		df = df.drop(['id', 'label', 'lane_Two_firstPoint', 'lane_Two_midPoint', 'lane_Two_lastPoint',
					  'lane_Three_firstPoint', 'lane_Three_midPoint', 'lane_Three_lastPoint'], axis=1)
		# Label Encoding on car_direction
		mapping_direc = {'': 0, 'F': 1, 'R': 2, 'L': 3}
		df['car_direction'] = df['car_direction'].map(mapping_direc)
		# Add each columns in one row as a list
		df = df.tail(12)

		df2 = pd.DataFrame(columns=df.columns)
		for i in df.columns:
			listIs = []
			for j in df[i]:
				listIs.append(j)
			df2[i] = listIs
			df2.iat[0, df2.columns.get_loc(i)] = str(listIs)
		df2 = df2.head(1)
		# df2.reset_index(inplace=True)
		df2['case'] = 2
		df2 = df2.drop(['index'], axis=1)
		# print(df2.columns)

		df2.to_csv('Test-new-all-cases.csv', mode='a', index=False, header=False)

