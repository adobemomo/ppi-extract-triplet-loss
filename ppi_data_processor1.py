# convert ddi xml into data

# data: [sent_id, sent_text, pair_list]
# pair_list: [entity1, entity2, ddi]
# ddi: true/false

import os
import pickle
import xml.etree.ElementTree as ET

from utils import *
import logger
import time
import sys
import os
sys.stdout = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.output', sys.stdout)
sys.stderr = logger.Logger('logs/'+time.strftime('%Y-%m-%d %H.%M.%S',time.localtime(time.time()))+os.path.basename(__file__)+'.error', sys.stderr)		# redirect std err, if necessary



def read_ppi_xml(dir):
    data = []
    file_list = os.listdir(dir)
    for fname in file_list:
        parser = ET.XMLParser(encoding="UTF-8")  # etree.XMLParser(recover=True)
        tree = ET.parse(dir + '/' + fname, parser=parser)
        root = tree.getroot()
        sent_cnt = 0
        pos_cnt = 0
        neg_cnt = 0

        for document in root:
            for sentence in document:
                sent_cnt += 1
                sentence_context = sentence.attrib['text']
                sentence_id = sentence.attrib['id']
                entity_dict = {}
                pos_interaction_list = []
                neg_interaction_list = []

                e_count = 0
                e_pair_flag = {}
                e_prefix = sentence_id + '.e'

                for item in sentence:
                    if item.tag == 'entity':
                        e_type = item.attrib['type']
                        e_id = item.attrib['id']
                        e_ch_offset = item.attrib['charOffset']
                        e_text = item.attrib['text']
                        entity_dict[e_id] = [e_text, e_type, e_ch_offset]
                        e_pair_flag[e_count] = []
                        e_count += 1

                for item in sentence:
                    if item.tag == 'interaction':
                        e1 = item.attrib['e1']
                        entity1 = entity_dict[e1]
                        e2 = item.attrib['e2']
                        entity2 = entity_dict[e2]
                        i_type = item.attrib['type']
                        pos_interaction_list.append([entity1, entity2, 'true'])
                        e1_int = int(e1[e1.rfind('e') + 1:])
                        e2_int = int(e2[e2.rfind('e') + 1:])
                        e_pair_flag[e1_int].append(e2_int)
                        e_pair_flag[e2_int].append(e1_int)

                # add negative instance into list
                for i in range(e_count):
                    for j in range(e_count):
                        if i not in e_pair_flag[j] or j not in e_pair_flag[i]:
                            entity1 = entity_dict[e_prefix + str(i)]
                            entity2 = entity_dict[e_prefix + str(j)]
                            i_type = "false"
                            neg_interaction_list.append([entity1, entity2, i_type])

                # if len(pos_interaction_list):
                #     print("neg_interaction_list", neg_interaction_list[0])
                #     print("pos_interaction_list", pos_interaction_list[0])


            # data.append([sentence_id, sentence_context, pos_interaction_list, neg_interaction_list])
                pos_cnt += len(pos_interaction_list)
                neg_cnt += len(neg_interaction_list)
                interaction_list = pos_interaction_list
                interaction_list.extend(neg_interaction_list)
                # if interaction_list is not None:
                data.append([sentence_id, sentence_context, interaction_list])
        print(fname[:-4], ": sent cnt=", sent_cnt, ", pos cnt=", pos_cnt, ", neg cnt=", neg_cnt)
    print("total sentence cnt=", len(data))
    return data


ppi_xml_dir = 'corpus/ppi_corpus'
ppi_step1_txt = 'dataset/ppi_data/step1/train.txt'
ppi_step1_pickle = 'dataset/ppi_data/step1/train.pickle'
ppi_data = read_ppi_xml(ppi_xml_dir)
write_step1_data_as_txt(ppi_data, ppi_step1_txt)
pickle.dump(ppi_data, open(ppi_step1_pickle, 'wb'))
# print(pickle.load(open(ddi_step1_pickle, 'rb')))
