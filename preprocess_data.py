import random
import xml.etree.ElementTree as ET
import pickle
import os

###################3
#调试用
##################3

# convert xml file to txt file

# read data from xml file
def read_ppi(filename):
    data = []

    parser = ET.XMLParser(encoding="UTF-8")  # etree.XMLParser(recover=True)
    tree = ET.parse(filename, parser=parser)
    root = tree.getroot()

    for document in root:
        for sentence in document:
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
                elif item.tag == 'interaction':
                    e1 = item.attrib['e1']
                    entity1 = entity_dict[e1]
                    e2 = item.attrib['e2']
                    entity2 = entity_dict[e2]
                    i_type = item.attrib['type']
                    pos_interaction_list.append([entity1, entity2, i_type])
                    e1_int = int(e1[e1.rfind('e') + 1:])
                    e2_int = int(e2[e2.rfind('e') + 1:])
                    e_pair_flag[e1_int].append(e2_int)
                    e_pair_flag[e2_int].append(e1_int)

            # add negative instance into list
            for i in range(e_count):
                for j in range(e_count):
                    if j in e_pair_flag[i]:
                        entity1 = entity_dict[e_prefix + str(i)]
                        entity2 = entity_dict[e_prefix + str(j)]
                        i_type = "false"
                        neg_interaction_list.append([entity1, entity2, i_type])

            # if len(pos_interaction_list):
            #     print("neg_interaction_list", neg_interaction_list[0])
            #     print("pos_interaction_list", pos_interaction_list[0])

            data.append([sentence_id, sentence_context, pos_interaction_list, neg_interaction_list])
    return data

# data: [sent_id, sent_text, pair_list]
# pair_list: [entity1, entity2, ddi]
# ddi: true/false
def read_ddi(dir):
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
                    # if ddi == 'true':
                    #     if 'type' in c.attrib:
                    #         ddi = c.attrib['type']
                    #     else:
                    #         ddi = 'int'
                    pair_list.append([entity1, entity2, ddi])
                    num_pair = num_pair + 1

            data.append([sent_id, sent_text, pair_list])
    print("num_pair:", num_pair)
    return data


def write(filename, data):
    with open(filename, mode='wb') as fw:
        pickle.dump(data, fw, 0)

def replace_ddi(data):
    count = 0
    replaced_data = []
    for s in data:
        sent = s[1]
        pair = s[2]

        e_dict = []

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            if d1 not in e_dict:
                e_dict.append(d1_offset)
            if d2 not in e_dict:
                e_dict.append(d2_offset)

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            ddi = p[2]
            if d1 == d2:
                continue
            count += 1

            if d1_offset.find(';') == -1 and d2_offset.find(';') == -1:
                d1_start, d1_end = d1_offset.split('-')
                d2_start, d2_end = d2_offset.split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)
                d2_start, d2_end = int(d2_start), int(d2_end)
            elif d1_offset.find(';') > -1 and d2_offset.find(';') > -1:
                d1_start, d1_end = d1_offset.split(';')[0].split('-')
                d2_start, d2_end = d2_offset.split(';')[0].split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)
                d2_start, d2_end = int(d2_start), int(d2_end)
            elif d1_offset.find(';') > -1 and d2_offset.find(';') == -1:
                d1_1, d1_2 = d1_offset.split(';')
                d1_1_start, d1_1_end = d1_1.split('-')
                d1_2_start, d1_2_end = d1_2.split('-')
                d1_1_start, d1_1_end = int(d1_1_start), int(d1_1_end)
                d1_2_start, d1_2_end = int(d1_2_start), int(d1_2_end)
                d2_start, d2_end = d2_offset.split('-')
                d2_start, d2_end = int(d2_start), int(d2_end)

                if len(set(range(d1_1_start, d1_1_end)) & set(range(d2_start, d2_end))):
                    d1_start, d1_end = d1_2_start, d1_2_end
                else:
                    d1_start, d1_end = d1_1_start, d1_1_end
            else:
                d2_1, d2_2 = d2_offset.split(';')
                d2_1_start, d2_1_end = d2_1.split('-')
                d2_2_start, d2_2_end = d2_2.split('-')
                d2_1_start, d2_1_end = int(d2_1_start), int(d2_1_end)
                d2_2_start, d2_2_end = int(d2_2_start), int(d2_2_end)
                d1_start, d1_end = d1_offset.split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)

                if len(set(range(d2_1_start, d2_1_end)) & set(range(d1_start, d1_end))):
                    d2_start, d2_end = d2_2_start, d2_2_end
                else:
                    d2_start, d2_end = d2_1_start, d2_1_end

            other = set(e_dict) - set([d1_offset, d2_offset])
            replaced_sent = sent.replace(sent[d1_start:d1_end+1], 'DRUGA').replace(sent[d2_start:d2_end+1], 'DRUGB')
            for n in other:
                if n.find(';') > -1:
                    n = n.split(';')[0]
                n_start, n_end = n.split('-')
                n_start, n_end = int(n_start), int(n_end)
                replaced_sent = replaced_sent.replace(sent[n_start:n_end+1], 'DRUGN')

            # print(replaced_sent)
            # print(sent)
            replaced_data.append([replaced_sent, d1, d1_type, d2, d2_type, ddi])

    print(replaced_data)
    return replaced_data

def replace_ddi_triplet(data):
    count = 0
    replaced_data = []
    for s in data:
        sent = s[1]
        pair = s[2]

        e_dict = []
        homo_negs = []

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            if d1 not in e_dict:
                e_dict.append(d1_offset)
            if d2 not in e_dict:
                e_dict.append(d2_offset)
            if ddi == 'false':
                homo_negs.append(p)

        for p in pair:
            d1, d1_type, d1_offset = p[0]
            d2, d2_type, d2_offset = p[1]
            ddi = p[2]
            if d1 == d2:
                continue
            count += 1

            if ddi == 'true' and len(homo_negs)>0:
                index = random.randint(0, len(homo_negs) - 1)
                homo_neg = homo_negs[index]

            if d1_offset.find(';') == -1 and d2_offset.find(';') == -1:
                d1_start, d1_end = d1_offset.split('-')
                d2_start, d2_end = d2_offset.split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)
                d2_start, d2_end = int(d2_start), int(d2_end)
            elif d1_offset.find(';') > -1 and d2_offset.find(';') > -1:
                d1_start, d1_end = d1_offset.split(';')[0].split('-')
                d2_start, d2_end = d2_offset.split(';')[0].split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)
                d2_start, d2_end = int(d2_start), int(d2_end)
            elif d1_offset.find(';') > -1 and d2_offset.find(';') == -1:
                d1_1, d1_2 = d1_offset.split(';')
                d1_1_start, d1_1_end = d1_1.split('-')
                d1_2_start, d1_2_end = d1_2.split('-')
                d1_1_start, d1_1_end = int(d1_1_start), int(d1_1_end)
                d1_2_start, d1_2_end = int(d1_2_start), int(d1_2_end)
                d2_start, d2_end = d2_offset.split('-')
                d2_start, d2_end = int(d2_start), int(d2_end)

                if len(set(range(d1_1_start, d1_1_end)) & set(range(d2_start, d2_end))):
                    d1_start, d1_end = d1_2_start, d1_2_end
                else:
                    d1_start, d1_end = d1_1_start, d1_1_end
            else:
                d2_1, d2_2 = d2_offset.split(';')
                d2_1_start, d2_1_end = d2_1.split('-')
                d2_2_start, d2_2_end = d2_2.split('-')
                d2_1_start, d2_1_end = int(d2_1_start), int(d2_1_end)
                d2_2_start, d2_2_end = int(d2_2_start), int(d2_2_end)
                d1_start, d1_end = d1_offset.split('-')
                d1_start, d1_end = int(d1_start), int(d1_end)

                if len(set(range(d2_1_start, d2_1_end)) & set(range(d1_start, d1_end))):
                    d2_start, d2_end = d2_2_start, d2_2_end
                else:
                    d2_start, d2_end = d2_1_start, d2_1_end

            other = set(e_dict) - set([d1_offset, d2_offset])
            replaced_sent = sent.replace(sent[d1_start:d1_end + 1], 'DRUGA').replace(sent[d2_start:d2_end + 1],
                                                                                     'DRUGB')
            for n in other:
                if n.find(';') > -1:
                    n = n.split(';')[0]
                n_start, n_end = n.split('-')
                n_start, n_end = int(n_start), int(n_end)
                replaced_sent = replaced_sent.replace(sent[n_start:n_end + 1], 'DRUGN')

            # print(replaced_sent)
            # print(sent)
            replaced_data.append([replaced_sent, d1, d1_type, d2, d2_type, ddi])

    print(replaced_data)
    return replaced_data

# data_xml = "data/xml/bioinfer-1.2.0b-unified-format.xml"
# step1_train_data = read(data_xml)
# # write("data/step1/train_data.txt", step1_train_data)
# pickle.dump(step1_train_data, open('data/step1/train_data.txt', 'wb'))
# data = pickle.load(open('data/step1/train_data.txt', 'rb'))
# print(data)

corpora = "ddi_data/xml/Train"
tr_data = read_ddi(corpora)
step2_tr_data = replace_ddi(tr_data)


