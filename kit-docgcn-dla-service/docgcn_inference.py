import pytesseract
import numpy as np
from PIL import Image

# IMPORTING Bottom-up Attention with Detectron2 - PYTORCH 
import os
import re

import detectron2
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from torch import nn


from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

# import some common libraries
import cv2
import torch

# Importing the libraries needed
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import BertTokenizer
#from transformers.configuration_bert import BertConfig
from torch import cuda
import logging
import nltk
# import benepar
import spacy
# from benepar.spacy_plugin import BeneparComponent
# import benepar


logging.basicConfig(level=logging.ERROR)

from class_utils import TextBERTEncoder, Feature_Extractor, RobertaClass, DocGCN_Encoder

print(f"DETECTRON VERSION : {detectron2.__version__}")

nltk.download('punkt')
# benepar.download('benepar_en3')
nlp = spacy.load('en_core_web_sm')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 100
INF_BATCH_SIZE = 1

# os.chdir('py-bottom-up-attention') 
cfg = get_cfg()
cfg.OUTPUT_DIR = "models"
cfg.merge_from_file("py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)
device = 'cuda' if cuda.is_available() else 'cpu'



def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def object_detection(raw_image, raw_boxes):
        # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes))
    # raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)
        
        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        print(pred_class_logits.shape)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        
        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes
        )
        
        return instances, roi_features


def visual_extraction(inf_image):
    actual_boxes = []
    inf_boxes = []
    inf_image = inf_image.convert("RGB")
    width, height = inf_image.size
    w_scale = 1000/width
    h_scale = 1000/height
    

    ocr_df = pytesseract.image_to_data(inf_image, output_type='data.frame') \
                
    ocr_df = ocr_df.dropna() \
                .assign(left_scaled = ocr_df.left*w_scale,
                        width_scaled = ocr_df.width*w_scale,
                        top_scaled = ocr_df.top*h_scale,
                        height_scaled = ocr_df.height*h_scale,
                        right_scaled = lambda x: x.left_scaled + x.width_scaled,
                        bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    inf_words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
        actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 
        actual_boxes.append(actual_box)

    for box in actual_boxes:
        inf_boxes.append(normalize_box(box, width, height))

    # Generating the bbox 
    inf_img_bbox_list = {}
    inf_img_bbox_list["texts"] = inf_words
    inf_img_bbox_list["bboxes"] = inf_boxes


    path = "data/testing_data/images/83443897.png"
    im = cv2.imread(path)
    instances, features = object_detection(im, np.array(inf_img_bbox_list["bboxes"]))
    visual_features = features.tolist()

    print("Visual features generated")

    return visual_features, inf_words, inf_boxes, actual_boxes


def embeddings_generator(data_loader, model_bert):
    output = []
    model_bert.eval()
    for _,data in tqdm(enumerate(data_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        # targets = data['targets'].to(device, dtype = torch.long)
        outputs = model_bert(ids, mask,token_type_ids)
        output.extend(outputs.tolist())

    print("Textual features generated")
    return output


def funsd_scene_graph_generator(input_path, type, inf_df):

  # There are two types scene graphs one is position based global scene graph
  # Another one is document components structural relationship based binary scene graph
  # Type 1 means global relationship which will return reletively globally positional relationship 
  # Type 2 will return binary structure or logical relationship between components

  json_path = input_path
  img_path = input_path
  img_list = os.listdir(img_path)
  # for training dataset bbox_id = 0 setting this to order all trainind and validation dataset
  # bbox_id = 7412 
  bbox_id = 0
  relation_id = 0
  bi_relation = 8473
  scene_graph = {}
  for index in range(len(img_list)):
    print(index)
    name = img_list[index][:-4]
    print(name)
    image = Image.open(img_path+'/'+name+'.png')
    # info_path = json_path+name+'.json'
    # with open(info_path) as f:
    #   info_list = json.load(f)
    # info_list = info_list['form']
    scene_graph[name] = {}
    scene_graph[name]['file_name'] = name
    scene_graph[name]['width'] = image.size[0]
    scene_graph[name]['height'] = image.size[1]
    scene_graph[name]['objects'] = {}
    for i, row in inf_df.iterrows():
      scene_graph[name]['objects'][str(i)] = {}
      scene_graph[name]['objects'][str(i)]['id'] = bbox_id
      bbox_id += 1
      scene_graph[name]['objects'][str(i)]['box'] = row['box']
      # scene_graph[name]['objects'][str(i)]['category'] = info_list[i]['label']
      scene_graph[name]['objects'][str(i)]['text'] = row['text']
      scene_graph[name]['objects'][str(i)]['relations'] = {}

    # returning scene graph is position based global scene graph
    if type == 1:
      for i in scene_graph[name]['objects']:
        current_box = scene_graph[name]['objects'][i]
        box1 = current_box['box']
        c_relation = 0
        for j in scene_graph[name]['objects']:
          if i != j:
            box2 = scene_graph[name]['objects'][j]['box']
            current_box['relations'][str(c_relation)] = {}
            current_box['relations'][str(c_relation)]['id'] = relation_id
            current_box['relations'][str(c_relation)]['object'] = scene_graph[name]['objects'][j]['id']
            # be cautious of ignoring this
            # current_box['relations'][str(c_relation)]['name'] = relative_position(box1,box2)
            relation_id += 1
            c_relation += 1


    # type 0 will return binary relationships between bounding box
    # for funsd dataset, it will represent the question and corresponding answer relationships
    # if current node is a question, the relation will be annotated as 'answer' to its answer
    # if current node is a answer, the relation will be annotated as 'question' 
    elif type == 0:
      for i in scene_graph[name]['objects']:
        current_box = scene_graph[name]['objects'][i]
        k = int(i)
        if len(info_list[k]['linking']) != 0:
          for n in range(len(info_list[k]['linking'])):
            linking_node = info_list[k]['linking'][n]
            current_box['relations'][str(n)] = {}
            node1 = linking_node[0]
            node2 = linking_node[1]
            if node1 == k:
              current_box['relations'][str(n)]['name'] = info_list[node2]['label']
              current_box['relations'][str(n)]['id'] = bi_relation
              current_box['relations'][str(n)]['object'] = scene_graph[name]['objects'][str(node2)]['id']
              bi_relation += 1
            elif node2 == k:
              current_box['relations'][str(n)]['name'] = info_list[node1]['label']
              current_box['relations'][str(n)]['id'] = bi_relation
              current_box['relations'][str(n)]['object'] = scene_graph[name]['objects'][str(node1)]['id']
              bi_relation += 1

  if type == 1:
    return scene_graph
  if type == 0:
    print(bbox_id,bi_relation)
    return scene_graph

# calculate gap distance between 
def gap_distance(bbox1, bbox2):

    x1, y1, x12, y12 = bbox1
    x2, y2, x22, y22 = bbox2
    w1 = abs(x12-x1)
    h1 = abs(y12-y1)
    w2 = abs(x22-x2)
    h2 = abs(y22-y2)
    dist = 0
    if x2>x1 and abs(x2-x1)>w1:
        dist = abs(x2-x1)-w1
    elif x2<x1 and abs(x2-x1)>w2:
        dist = abs(x2-x1)-w2
    elif y1>y2 and abs(y1-y2)>h2:
        dist = abs(y2-y1)-h2
    elif y2>y1 and abs(y1-y2)>h1:
        dist = abs(y2-y1)-h1
    return dist

# number of characters / bounding box size
def char_density(obj):
    text = obj['text']
    x1,y1,x2,y2 = obj['box']
    size = abs(x2-x1)*abs(y2-y1)
    density = len(text)/size
    obj['char_density'] = density

# number of characters
def char_number(obj):
    text = obj['text']
    char_num = len(text)
    obj['char_number'] = char_num


# number of tokens / bounding box size
def token_density(obj):
    text = obj['text']
    x1,y1,x2,y2 = obj['box']
    tokenized_text = nltk.word_tokenize(text)
    size = abs(x2-x1)*abs(y2-y1)
    density = len(tokenized_text)/size
    obj['text_density'] = density
    

# number of tokens
def token_number(obj):
    text = obj['text']
    tokenized_text = nltk.word_tokenize(text)
    token_num = len(tokenized_text)
    obj['text_number'] = token_num


def syntactic_parser(inference_scenegraph_positional):
   
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})


    for img in inference_scenegraph_positional:
        print(img)
        if img not in ['92314414']:
            for i in inference_scenegraph_positional[img]['objects']:
                box = inference_scenegraph_positional[img]['objects'][i]
                text = box['text']
                doc = nlp(text)
                try:
                    sent = list(doc.sents)[0]
                    parse = sent._.parse_string
                    _,level1,level2 = constituency_parsing_extractor(parse)
                    box['parsing_level1'] = level1
                    box['parsing_level2'] = level2
                except:
                    box['parsing_level1'] = []
                    box['parsing_level2'] = []
        else:
            for x in inference_scenegraph_positional[img]['objects']:
                box = inference_scenegraph_positional[img]['objects'][x]
                box['parsing_level1'] = []
                box['parsing_level2'] = []
    print("Syntactic parser completed")

    return inference_scenegraph_positional
   

def test_label_generator(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    output_list = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            visual_feature = data['visual_feature'].to(device, dtype = torch.float)
            gcn_bert_base = data['gcn_bert_base'].to(device, dtype = torch.float)
            parsing1 = data['parsing1'].to(device, dtype = torch.float)
            parsing2 = data['parsing2'].to(device, dtype = torch.float)
            char_density = data['char_density'].to(device, dtype = torch.float)
            char_number = data['char_number'].to(device, dtype = torch.float)
            # pos_emb = data['pos_emb'].to(device, dtype = torch.float)
            visual = data['visual'].to(device, dtype = torch.float)

            # outputs = model(ids, mask, token_type_ids,visual_feature,char_density,char_number,parsing1,parsing2,gcn_bert_base,visual).squeeze()
            outputs = model(ids, mask, token_type_ids,char_density,char_number, visual_feature,gcn_bert_base,parsing1,parsing2,visual)
            
            big_val, big_idx = torch.max(outputs.data, dim=1)
            output_list = output_list + list(big_idx)

            nb_tr_steps += 1
            
    return output_list


def inference_main(inf_image):
   
    visual_features, ocr_texts, ocr_boxes, actual_boxes  = visual_extraction(inf_image)
    inf_df = pd.DataFrame.from_dict({"text":ocr_texts, "box":ocr_boxes, "raw_bbox":actual_boxes})
    inf_df["id"] = [i for i in range(inf_df.shape[0])]

    sample_size = 1
    inf_data = inf_df.sample(frac = sample_size,random_state = 200)
    inf_data = inf_data.reset_index(drop=True)

    inference_set = TextBERTEncoder(inf_data, tokenizer, MAX_LEN)
    inference_params = {'batch_size': INF_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }
    inference_loader = DataLoader(inference_set, **inference_params)
    # model_bert = torch.load('models/funsd_bert_embedder_v3.pt')
    model_bert = Feature_Extractor()
    model_bert.load_state_dict(torch.load('models/funsd_bert_embedder_v4.pt'))
    model_bert.to(device)# Defining the training function on the 80% of the dataset for tuning the distilbert model

    text_embeddings = embeddings_generator(inference_loader, model_bert)
    inf_df["visual_embeddings"] = visual_features
    inf_df["bert_large_emb"] = text_embeddings

    if os.path.isfile('data/inference_data/.DS_Store'):
       os.remove('data/inference_data/.DS_Store')

    inference_scenegraph_positional = funsd_scene_graph_generator('data/inference_data/', 1, inf_df)

    for l in inference_scenegraph_positional:
        obj = inference_scenegraph_positional[l]['objects']
        for obj1 in obj:
            obj[obj1]['gap'] = {}
            for obj2 in obj:
                if obj1 != obj2:
                    dist = gap_distance(obj[obj1]['box'],obj[obj2]['box'])
                    obj[obj1]['gap'][obj2] = dist

    
    for l in inference_scenegraph_positional:
        for obj in inference_scenegraph_positional[l]['objects']:
            token_density(inference_scenegraph_positional[l]['objects'][obj])
            token_number(inference_scenegraph_positional[l]['objects'][obj])
            char_density(inference_scenegraph_positional[l]['objects'][obj])
            char_number(inference_scenegraph_positional[l]['objects'][obj])

    # inference_scenegraph_positional = syntactic_parser(inference_scenegraph_positional)

    # model_docgcn = torch.load('models/funsd_docgcn_bert_layout_segment_classifier.pt')
    model_docgcn = RobertaClass()
    model_docgcn.load_state_dict(torch.load('models/funsd_docgcn_bert_layout_segment_classifier_v2.pt'))
    model_docgcn.to(device)# Defining the training function on the 80% of the dataset for tuning the distilbert model

    inference_list = []
    for file in inference_scenegraph_positional:
        for x in inference_scenegraph_positional[file]['objects']:
            inference_list.append(inference_scenegraph_positional[file]['objects'][x])

    df_infer = pd.DataFrame(inference_list)
    df_merged = pd.merge(df_infer, inf_df[['id','visual_embeddings','bert_large_emb','raw_bbox']]\
                    , on=['id','id'], how='left') 
    
    df_sample = pd.read_csv("funsd/df_train_docgcn_final.csv")
    df_sample['gcn_token_number'] = df_sample['gcn_token_number'].apply(lambda x: re.findall(r"[-+]?(?:\d*\.*\d+)", x))
    df_sample['gcn_token_number'] = df_sample['gcn_token_number'].apply(lambda x: list(map(float, x)))

    df_merged["gcn_token_number"]=[df_sample["gcn_token_number"].loc[5]]*len(df_merged)

    df_merged["parsing_level1"] = df_merged.apply(lambda x: [], axis=1)

    df_merged["parsing_level2"] = df_merged.apply(lambda x: [], axis=1)

    cols_left  = ['text','near_visual_feature','gcn_near_char_density',\
                   'gcn_near_char_number','level1_parse_emb','level2_parse_emb',\
                   'density','visual_feature','gcn_bert_predicted']

    cols_right = ['text', 'gcn_token_number', 'char_density',\
                        'char_number', 'parsing_level1', 'parsing_level2',\
                        'text_density', 'visual_embeddings', 'bert_large_emb']
    
    new_inf_df = pd.DataFrame()
    new_inf_df[cols_left] = df_merged[cols_right]

    inf_params = {'batch_size': INF_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

    test = DocGCN_Encoder(new_inf_df, tokenizer, MAX_LEN)
    inference_loader = DataLoader(test, **inf_params)

    output = test_label_generator(model_docgcn, inference_loader)

    q = []
    for p in output:
        q.append(p.cpu().numpy().tolist())
    # print(q)
    iob_to_label = {3:'question', 0: 'answer', 2: 'other', 1: 'header'}

    new_inf_df["label_predictions"] = q
    new_inf_df["text_predictions"] = new_inf_df["label_predictions"].map(iob_to_label)

    new_inf_df["bbox"] = df_merged["raw_bbox"]

    cols_res  = ['text','bbox','gcn_near_char_density',\
                   'gcn_near_char_number','level1_parse_emb','level2_parse_emb',\
                   'density', 'label_predictions', 'text_predictions']

    results = new_inf_df[cols_res].to_dict(orient='records')
    return results


# if __name__ == '__main__':

#     start_inf_time = time.time()
#     inf_image = Image.open("data/testing_data/images/83443897.png")
#     device = 'cuda' if cuda.is_available() else 'cpu'
#     inference_main(inf_image)
#     stop_inf_time = time.time()
#     print("--- %s seconds ---" % (stop_inf_time - start_inf_time))
    



    

    


