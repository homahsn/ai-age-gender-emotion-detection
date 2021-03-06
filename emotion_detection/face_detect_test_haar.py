import cv2
import sys
import csv

face_rects = [
[512,216,242,327],
[437,174,171,272],
[584,593,57,73],
[357,191,284,374],
[335,28,431,583],
[699,105,182,271],
[331,96,211,284],
[122,65,66,88],
[372,179,353,474],
[535,104,115,165],
[66,267,382,507],
[481,139,100,137],
[288,62,169,223],
[102,-5,828,1082],
[486,292,183,224],
[41,143,217,294],
[354,172,409,548],
[324,126,401,565],
[278,81,120,158],
[423,188,235,298],
[316,115,434,573],
[385,108,113,147],
[264,175,488,674],
[192,-3,336,422],
[388,90,245,332],
[413,408,121,149],
[342,-3,443,581],
[550,191,185,259],
[247,263,249,310],
[4,-1,1077,1076],
[338,126,396,569],
[199,95,547,724],
[398,338,120,192],
[378,286,223,315],
[352,39,168,207],
[428,272,312,401],
[275,153,413,451],
[215,81,218,285],
[390,223,426,541],
[154,187,300,373],
[515,107,142,194],
[321,303,295,398],
[553,179,91,121],
[317,141,530,661],
[363,96,140,191],
[273,64,564,795],
[285,412,150,186],
[537,190,139,189],
[440,193,279,353],
[261,327,478,616],
[334,750,132,166],
[212,284,300,378],
[474,113,158,198],
[342,220,53,76],
[385,156,175,235],
[461,83,97,140],
[373,64,95,118],
[384,139,213,266],
[227,146,32,47],
[397,69,241,316],
[259,232,430,576],
[232,96,479,631],
[164,4,562,693],
[375,237,96,116],
[426,86,156,199],
[284,309,487,631],
[380,498,77,98],
[249,195,238,346],
[333,160,124,158],
[579,576,68,93],
[507,143,221,275],
[395,205,309,415],
[284,341,353,471],
[324,108,277,393],
[213,173,383,461],
[481,77,199,224],
[518,189,83,119],
[223,38,163,218],
[339,163,274,372],
[120,72,621,751],
[306,260,241,312],
[362,204,313,420],
[492,298,59,74],
[443,299,367,534],
[357,401,154,234],
[328,169,347,477],
[295,120,322,353],
[479,131,219,281],
[477,116,461,638],
[542,303,96,120],
[109,105,118,146],
[339,5,251,355],
[277,182,446,608],
[308,69,126,147],
[378,239,356,438],
[209,188,307,362],
[435,118,243,343],
[270,242,318,421],
[641,276,115,153],
[262,113,546,720],
[303,105,176,205],
[700,172,113,135],
[493,93,124,167],
[658,169,108,147],
[592,350,127,186],
[329,172,391,543],
[476,270,216,279],
[482,313,405,542],
[326,163,108,139],
[441,472,123,159],
[335,111,98,135],
[520,344,92,123],
[203,96,84,109],
[272,73,323,427],
[339,440,302,381],
[377,207,244,343],
[517,207,131,183],
[81,111,230,284],
[528,439,384,501],
[495,181,186,245],
[495,224,99,135],
[286,333,435,590],
[504,107,283,346],
[445,229,233,290],
[349,143,69,89],
[89,97,122,168],
[375,144,385,492],
[372,146,221,268],
[265,107,141,204],
[386,436,289,370],
[416,117,417,559],
[380,122,151,200],
[392,412,181,210],
[387,391,220,298],
[412,105,271,355],
[562,191,234,243],
[138,109,542,724],
[3,2,720,1004],
[444,148,177,234],
[350,150,368,509],
[564,138,193,239],
[552,436,152,187],
[588,111,180,283],
[163,105,326,428],
[509,203,93,105],
[156,70,213,259],
[359,368,343,465],
[473,69,186,272],
[174,170,231,297],
[211,281,555,748],
[205,57,262,360],
[622,234,233,296],
[284,216,573,683],
[320,311,279,339],
[246,174,182,212],
[428,143,201,266],
[138,4,610,902],
[403,283,206,251],
[379,129,447,595],
[357,251,522,730],
[85,238,566,724],
[480,303,359,427],
[442,230,250,331],
[338,136,552,639],
[513,167,167,217],
[423,344,120,149],
[368,451,360,466],
[393,96,420,552],
[205,180,173,236],
[113,105,300,408],
[430,114,249,348],
[286,463,384,467],
[337,45,380,506],
[320,542,382,505],
[175,147,593,801],
[232,154,194,260],
[345,273,443,645],
[270,182,478,676],
[478,86,476,662],
[483,157,390,522],
[292,70,573,752],
[371,430,180,247],
[411,116,195,280],
[342,348,365,450],
[413,87,224,333],
[515,225,133,169],
[298,98,249,343],
[297,97,444,609],
[485,179,134,184],
[513,115,81,110],
[578,167,107,139],
[218,448,311,450],
[231,275,349,447],
[380,76,161,203],
[211,73,233,320],
[441,97,166,228],
[237,173,385,491],
[282,144,279,366],
[237,164,323,409],
[666,410,379,513]]

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
def get_face(path):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    image = cv2.imread(path)
    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,minNeighbors=10)#,minSize=(64,64),flags=cv2.CASCADE_SCALE_IMAGE)
    print("Found {0} faces!".format(len(faces)))
    rect = [0,0,width,height]
    for (x, y, w, h) in faces:
        rect=[x,y,w,h]
        break
    return rect

detected = []
for i in range(1,201):
	fn = "./insta_data/%03d.jpg"%(i)
	rect = get_face(fn)
	rr = face_rects[i-1]
	bb1 = {"x1":rr[0],"y1":rr[1],"x2":rr[0]+rr[2],"y2":rr[1]+rr[3]}
	bb2 = {"x1":rect[0],"y1":rect[1],"x2":rect[0]+rect[2],"y2":rect[1]+rect[3]}
	iou = get_iou(bb1,bb2)
	if iou>0.75:
		detected.append([True])
	else:
		detected.append([False])


with open('facedetect-haar-result.csv', 'w') as f: 
	csvwriter = csv.writer(f) 
    # writing the fields 
	csvwriter.writerows(detected)
