import cv2
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument("--image", "-i", type=str, default=None)
    parser.add_argument("--video", "-v", type=str, default=None)
    parser.add_argument("--webcam", "-w", type=int, default=0)
    parser.add_argument("--size", "-s", type=int, default=416)
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    parser.add_argument("--vid_out", type=str, default=None)
    args = parser.parse_args()
    return args

args = get_args()

def plot_boxes(results, frame, classes):
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    ### looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= args.threshold: ### threshold value for detection
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]

            if text_d == 'with_mask' or text_d == "mask_weared_incorrect":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                (w, h), _ = cv2.getTextSize("mask", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(frame, "mask", (x1, y1 - 5)   , cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255,255,255), 1, cv2.LINE_AA)
                
            elif text_d == 'without_mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) ## BBox
                (w, h), _ = cv2.getTextSize("no_mask", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (255, 0, 0), -1)
                cv2.putText(frame, "no_mask", (x1, y1 - 5)   , cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255,255,255), 1, cv2.LINE_AA)
    return frame


def main(image=args.image, vid_path=args.video, web_cam = args.webcam, vid_out=args.vid_out):

    model = torch.hub.load('Yolov5_model/yolov5','custom',path='best.pt',source='local')
    classes = model.names
    if image!=None:
        im = cv2.imread(image)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        results = model(im)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = plot_boxes(results, im , classes)
        cv2.imwrite('out_put.jpg',im)
        cv2.imshow('test',im)
        cv2.waitKey(0)
    if vid_path != None or web_cam != None:
        if vid_path != None:
            cap = cv2.VideoCapture(vid_path)
        else:
            cap = cv2.VideoCapture(web_cam)
        if vid_out:
            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'MJPG') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))
        cv2.namedWindow("out_put", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame, classes)
                cv2.imshow("out_put",frame)
                if vid_out:
                    out.write(frame)
                if cv2.waitKey(24) & 0xFF==ord("q"):
                    break
        cap.release()
        out.release()
        
        cv2.destroyAllWindows()
            
            

main()

