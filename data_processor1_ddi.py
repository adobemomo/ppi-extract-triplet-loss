# convert ddi xml into data

# data: [sent_id, sent_text, pair_list]
# pair_list: [entity1, entity2, ddi]
# ddi: true/false

import random
import xml.etree.ElementTree as ET
import pickle
import os
from utils import *


def read_ddi_xml(dir):
    data = []
    num_pair = 0
    print(dir)
    file_list = os.listdir(dir)
    for fname in file_list:
        parser = ET.XMLParser(encoding="UTF-8")  # etree.XMLParser(recover=True)
        tree = ET.parse(dir + '/' + fname, parser=parser)
        root = tree.getroot()
        for sent in root:
            sent_id = sent.attrib['id']
            sent_text = sent.attrib['text'].strip()
            ent_dict = {}
            pair_list = []
            for c in sent:
                # obtain entities' information
                if c.tag == 'entity':
                    d_type = c.attrib['type']
                    d_id = c.attrib['id']
                    d_ch_of = c.attrib['charOffset']
                    d_text = c.attrib['text']
                    ent_dict[d_id] = [d_text, d_type, d_ch_of]
                # obtain entity pairs' information
                elif c.tag == 'pair':
                    p_id = c.attrib['id']
                    e1 = c.attrib['e1']
                    entity1 = ent_dict[e1]
                    e2 = c.attrib['e2']
                    entity2 = ent_dict[e2]
                    ddi = c.attrib['ddi']

                    pair_list.append([entity1, entity2, ddi])
                    num_pair = num_pair + 1

            data.append([sent_id, sent_text, pair_list])
    print("num_pair:", num_pair)
    return data


ddi_xml_dir = 'corpus/ddi_corpus'
ddi_step1_txt = 'dataset/ddi_data/step1/train.txt'
ddi_step1_pickle = 'dataset/ddi_data/step1/train.pickle'
ddi_data = read_ddi_xml(ddi_xml_dir)
write_step1_data_as_txt(ddi_data, ddi_step1_txt)
pickle.dump(ddi_data, open(ddi_step1_pickle, 'wb'))
# print(pickle.load(open(ddi_step1_pickle, 'rb')))

