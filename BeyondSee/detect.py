"""
Usage:
    python detect.py --files "detect_input/1.jpg" --descriptor "sentiments" --phrase "active"
"""
from typing import List
import numpy as np
import cv2
import torch, torchvision
import pickle
import argparse
from preprocess import descriptors as desc
import time

# Add arguments to parser
parser = argparse.ArgumentParser(
    description="Detection of symbolic bounding boxes and labels"
)
parser.add_argument(
    "--files",
    dest="files",
    help="List of image files",
    default="None",
    type=lambda s: [item.lower() for item in s.split()],
)
parser.add_argument(
    "--descriptor", dest="descriptor", help="Descriptor", default="None", type=str
)
parser.add_argument(
    "--phrase", dest="phrase", help="Image phrase", default="None", type=str
)
parser.add_argument(
    "--threshold", dest="threshold", help="Detection threshold", default="0", type=str
)

# set the computation device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def detect(filelist: List[str], phrase: str, descriptor="sentiments", detection_threshold="0", test=False):
    """Generate image that contains predicted symbolic bounding boxes and labels
    and store under the detect_output directory

    Args:
        filelist (List[str]): a list of image file name
        phrase (str): phrase thatdescribes the images (refer to Sentiments_List.txt, Strategies_List.txt and Topics_List.txt)
        descriptor (str, optional): sentiments, strategies, or topics. Defaults to "sentiments".
        detection_threshold (str, optional): threshold for the predicted bounding boxes. Defaults to "0".
    """

    detection_threshold = float(detection_threshold)

    # TODO: Model filename to be modified if applicable
    if descriptor == "strategies":
        #model = "outputs/cp_strategies_tfasterrcnn_10CALS.pth.tar"
        model = "outputs/cp_strategies_tfasterrcnn_trip3ep.pth.tar"
    elif descriptor == "topics":
        model = "outputs/cp_topics_tfasterrcnn_10ep.pth.tar"
    # Default: sentiment model
    elif descriptor == "sentiments":
        model = "outputs/cp_sentiments_tfasterrcnn_10ep.pth.tar"
        #model = "outputs/cp_sentiments_tfasterrcnn_bsight.pth.tar"
    elif descriptor == "slogans":
        model = "outputs/cp_slogans_tfasterrcnn_10ep.pth.tar"
    elif descriptor == "qa":
        model = "outputs/cp_qa_tfasterrcnn_10ep.pth.tar"
    else:
        model = "outputs/cp_fasterrcnn.pth.tar"
    

    # Load model
    model_name = model
    model = torch.load(model)["model"]
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)

    # Evaluate model
    model.eval()

    # Label classes
    le = pickle.loads(open("outputs/le.pickle", "rb").read())
    CLASSES = le.classes_

    # Generate random color
    np.random.seed(10)  # seed value
    COLORS = [tuple(np.random.randint(256, size=3)) for _ in range(len(CLASSES))]
    COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in COLORS]

    if descriptor == "qa" and model_name == "outputs/cp_qa_roberta_tfasterrcnn_10ep.pth.tar":
        text_embed = desc.SentenceEmbedModel("stsb-roberta-base")
    else:
        text_embed = desc.TextEmbedModel()
    #   
    """
    if qa_roberta is used, text_embed needs to be sentence embed, else we can use text embed
    """
    #text_embed = desc.TextEmbedModel()

    phrase_embed = text_embed.get_vector_rep(phrase)
    phrase_embed = [torch.from_numpy(phrase_embed).float()]
    results = []
    for i in range(len(filelist)):
        # get the image file name for saving output later on
        image_name = filelist[i].split("/")[-1]
        
        image = cv2.imread(filelist[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = image.unsqueeze_(0)

        #idk just test
        with torch.no_grad():
            if descriptor == "original":
                outputs = model(image.to("cuda"))
            else:
                image_c = image.to("cuda")
                start_time = time.time()
                outputs = model(image_c, phrase_embed)
                end_time = time.time()
                print("Time Taken: " + str(end_time-start_time))
                

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        print(f"Length of outputs is {len(outputs[0]['boxes'])}")
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()

            #take top two scores
            # if len(boxes) > 2:
            #     one, two = get2max_index(scores)
            #     boxes = [boxes[one], boxes[two]]
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            draw_boxes = boxes.copy()

            # get all the predicted class names
            pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]
            i = 0

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                # find the index of the predicted label class
                index = np.where(CLASSES == pred_classes[j])[0][0]

                if not test:
                    cv2.rectangle(
                        orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        COLORS[index],
                        2,
                    )
                    draw_text(
                        img=orig_image,
                        text=pred_classes[j],
                        pos=(int(box[0]), int(box[1])),
                        text_color_bg=COLORS[index],
                    )
                else:
                    res = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), pred_classes[j], scores[i]]
                    results.append(res)
                    i += 1

            if not test:
                print(f"Writing {image_name} to file...")
                cv2.imwrite(f"detect_output/{image_name}", orig_image)
        if not test:
            print(f"Image {i+1}: {image_name} done...")
            print("-" * 50)

    cv2.destroyAllWindows()
    if not test:
        print("TEST PREDICTIONS COMPLETE")
    if test:
        return results


def draw_text(
    img,
    text,
    pos,
    text_color_bg,
    text_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    font_thickness=1,
):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h),
        font,
        font_scale,
        text_color,
        font_thickness
    )

def get2max_index(lst):
    if len(lst) <= 2:
        return None
    max1 = 0
    m1 = -1
    max2 = 0
    m2 = - 1
    for i in range(len(lst)):
        current = lst[i]
        localmax = current > m1
        localmax2 = current > m2
        if localmax and localmax2:
            max2 = max1
            m2 = m1
            max1 = i
            m1 = current
        elif localmax2:
            max2 = i
            m2 = current
    return (max1, max2)


if __name__ == "__main__":
    args = parser.parse_args()
    detect(filelist=args.files, phrase=args.phrase.lower(), descriptor=args.descriptor.lower(), detection_threshold=args.threshold)
