import numpy as np
import random
import re
import pickle
from nltk.tokenize import WordPunctTokenizer

tokenizer = WordPunctTokenizer()


def preProcess(sent):
    # print("000", sent)
    # sent = sent.lower()
    sent = sent.replace('/', ' / ')
    sent = sent.replace('.', ' . ')
    sent = sent.replace(',', ' , ')

    sent = sent.replace('(',' ( ')
    sent = sent.replace(')',' ) ')
    sent = sent.replace('[',' [ ')
    sent = sent.replace(']',' ] ')

    sent = sent.replace(':',' : ')
    sent = sent.replace(';',' ; ')
    sent = sent.replace('-', ' - ')

    sent = tokenizer.tokenize(sent)
    sent = ' '.join(sent)
    sent = re.sub('\d', ' digital ', sent)
    return sent


def find_sub_list(sl, l):
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            return ind, ind + sll - 1


def makePaddedList(sent_contents, maxl, pad_symbol='<pad>'):
    T = []
    for sent in sent_contents:
        t = []
        lenth = len(sent)
        for i in range(lenth):
            t.append(sent[i])
        for i in range(lenth, maxl):
            t.append(pad_symbol)
        T.append(t)

    return T


def makeWordList(lista):
    sent_list = sum(lista, [])
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0

    wl = []
    i = 1

    wl.append('<pad>')
    wl.append('<unkown>')
    for w, f in wf.items():
        wl.append(w)
    return wl


def makeDistanceList(lista):
    sent_list = sum(lista, [])
    wf = {}
    for sent in sent_list:
        for w in sent:
            if w in wf:
                wf[w] += 1
            else:
                wf[w] = 0

    wl = []
    i = 1
    for w, f in wf.items():
        wl.append(w)
    return wl


def makeWordListReverst(word_list):
    wl = {}
    v = 0
    for k in word_list:
        wl[v] = k
        v += 1
    return wl


def mapWordToId(sent_contents, word_list):
    T = []
    for sent in sent_contents:
        t = []
        for w in sent:
            t.append(word_list.index(w))
        T.append(t)
    return T


def mapLabelToId(sent_lables, label_dict):
    if len(label_dict) > 2:
        return [label_dict[label] for label in sent_lables]
    else:
        return [int(label != 'false') for label in sent_lables]


"""	
Given his tenuous respiratory status , he was transferred to the FICU with closer observation .
his tenuous respiratory status|1|4|problem
closer observation|13|14|test
TeCP
"""


def makeFeaturesCRE(fname):
    print("Reading data and Making features")
    fp = open(fname, 'r')
    samples = fp.read().strip().split('\n\n')

    sent_list = []  # 2-d array [[w1,w2,....] ...]
    sent_lables = []  # 1-d array
    p1_list = []
    p2_list = []
    type_list = []
    length_list = []
    for sample in samples:

        sent, entity1, entity2, relation = sample.strip().split('\n')
        # PreProcess
        sent = sent.lower()  # pre processing
        sent = re.sub('\d', 'dg', sent)  # Pre processing

        e1, e1_s, e1_e, e1_t = entity1.split('|')
        e2, e2_s, e2_e, e2_t = entity2.split('|')

        word_list = sent.split()
        word_1 = word_list[0:int(e1_s)]
        word_2 = word_list[int(e1_e) + 1:int(e2_s)]
        word_3 = word_list[int(e2_e) + 1:]
        words = word_1 + [e1_t] + word_2 + [e2_t] + word_3
        s1 = words.index(e1_t)
        s2 = words.index(e2_t)

        # distance1 feature
        p1 = []
        for i in range(len(words)):
            if i < s1:
                p1.append(str(i - s1))
            elif i > s1:
                p1.append(str(i - s1))
            else:
                p1.append('0')

        # distance2 feature
        p2 = []
        for i in range(len(words)):
            if i < s2:
                p2.append(str(i - s2))
            elif i > s2:
                p2.append(str(i - s2))
            else:
                p2.append('0')

        # type feature
        t = []
        for i in range(len(words)):
            t.append('Out')
        t[s1] = e1_t
        t[s2] = e2_t

        sent_lables.append(relation)
        sent_list.append(words)
        p1_list.append(p1)
        p2_list.append(p2)
        type_list.append(t)
        length_list.append(len(words))

    return sent_list, p1_list, p2_list, type_list, length_list, sent_lables



def dataRead(fname):
    print("Input File Reading")
    fp = open(fname, 'r')
    samples = fp.read().strip().split('\n\n')
    sent_lengths = []  # 1-d array
    sent_contents = []  # 2-d array [[w1,w2,....] ...]
    sent_lables = []  # 1-d array
    entity1_list = []  # 2-d array [[e1,e1_t] [e1,e1_t]...]
    entity2_list = []  # 2-d array [[e1,e1_t] [e1,e1_t]...]
    for sample in samples:
        sent, entities, relation = sample.strip().split('\n')
        #		if len(sent.split()) > 100:
        #			continue
        e1, e1_t, e2, e2_t = entities.split('\t')
        sent_contents.append(sent.lower())
        entity1_list.append([e1, e1_t])
        entity2_list.append([e2, e2_t])
        sent_lables.append(relation)

    return sent_contents, entity1_list, entity2_list, sent_lables


def makeFeatures(sent_list, entity1_list, entity2_list):
    print('Making Features')
    word_list = []
    p1_list = []
    p2_list = []
    type_list = []

    for sent, ent1, ent2 in zip(sent_list, entity1_list, entity2_list):
        sent = preProcess(sent)
        # print(sent)
        sent_list1 = sent.split()
        # entity1 = preProcess(ent1[0]).split()
        # entity2 = preProcess(ent2[0]).split()
        try:
            try:
                s1 = sent_list1.index('proteina')
            except:
                s1 = sent_list1.index('proteinas')

            try:
                s2 = sent_list1.index('proteinb')
            except:
                s2 = sent_list1.index('proteinbs')
        except:
            print("????", sent)
            print(ent1, ent2)
            continue
        # distance1 feature
        p1 = []
        for i in range(len(sent_list1)):
            if i < s1:
                p1.append(str(i - s1))
            elif i > s1:
                p1.append(str(i - s1))
            else:
                p1.append('0')
        # distance2 feature
        p2 = []
        for i in range(len(sent_list1)):
            if i < s2:
                p2.append(str(i - s2))
            elif i > s2:
                p2.append(str(i - s2))
            else:
                p2.append('0')
        # type feature 'out out out type out out type'
        t = []
        for i in range(len(sent_list1)):
            t.append('Out')
        # print(len(t), t)
        # print(s1, s2)
        t[s1] = ent1[1]
        t[s2] = ent2[1]

        word_list.append(sent_list1)
        p1_list.append(p1)
        p2_list.append(p2)
        type_list.append(t)

    return word_list, p1_list, p2_list, type_list


def readWordEmb(word_list, fname, embSize=100):
    print("Reading word vectors")
    wv = []
    wl = []
    with open(fname, 'r',encoding='utf-8') as f:
        for line in f:
            vs = line.split()
            # print("len of vs:", len(vs))
            if len(vs) < embSize:
                print("worp2vec_info:",line)
                continue
            vect = list(map(float, vs[1:]))
            wv.append(vect)
            wl.append(vs[0])
    wordemb = []
    count = 0
    for word in word_list:
        if word in wl:
            wordemb.append(wv[wl.index(word)])
        else:
            count += 1
            wordemb.append(np.random.rand(embSize))
        # wordemb.append( np.random.uniform(-np.sqrt(3.0/embSize), np.sqrt(3.0/embSize) , embSize) )

    wordemb[word_list.index('<pad>')] = np.zeros(embSize)
    print("wordembed_shape:", len(wordemb))

    wordemb = np.asarray(wordemb, dtype='float32')

    print("total number of word_list:", len(word_list))
    print("number of unknown word in word embepping:", count)
    return wordemb


def findLongestSent(Tr_word_list, Te_word_list):
    combine_list = Tr_word_list + Te_word_list
    a = max([len(sent) for sent in combine_list])
    return a


def findSentLengths(tr_te_list):
    lis = []
    for lists in tr_te_list:
        lis.append([len(l) for l in lists])
    return lis


def paddData(listL, maxl):  # W_batch, d1_tatch, d2_batch, t_batch)
    rlist = []
    for mat in listL:
        mat_n = []
        for row in mat:
            lenth = len(row)
            t = []
            for i in range(lenth):
                t.append(row[i])
            for i in range(lenth, maxl):
                t.append(0)
            mat_n.append(t)
        rlist.append(np.array(mat_n))
    return rlist


def makeBalence(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables):
    sent_contents = []
    entity1_list = []
    entity2_list = []
    sent_lables = []
    other = []
    clas = []
    for sent, e1, e2, lab in zip(Tr_sent_contents, Tr_entity1_list, Tr_entity2_list, Tr_sent_lables):
        if lab == 'false':
            other.append([sent, e1, e2, lab])
        else:
            clas.append([sent, e1, e2, lab])

    random.shuffle(other)

    neg = other[0: 3 * len(clas)]
    l = neg + clas
    for sent, e1, e2, lab in l:
        sent_contents.append(sent)
        entity1_list.append(e1)
        entity2_list.append(e2)
        sent_lables.append(lab)
    return sent_contents, entity1_list, entity2_list, sent_lables





# step1_data: [[sent_id, sent_text, [entity1, entity2, ddi]]]
def write_step1_data_as_txt(data, filename):
    fw = open(filename, 'w')
    for sid, stext, pair in data:
        if len(pair) == 0:
            continue
        fw.write(sid + "\t" + stext + "\n")
        for e1, e2, ppi in pair:
            fw.write(e1[0] + '\t' + e1[1] + '\t' + e1[2] + '\t' + e2[0] + '\t' + e2[1] + '\t' + e2[2] + '\t' + ppi)
            fw.write('\n')
        fw.write('\n')

# step2_data: [[sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]]
def write_step2_data_as_txt(data, filename):
    fw = open(filename, 'w')
    for id, dsent, p1, p1_type, p2, p2_type, ppi in data:
        fw.write(id + '\n')
        fw.write(dsent + '\n')
        fw.write(p1 + "\t" + p1_type + "\t" + p2 + "\t" + p2_type + '\n')
        fw.write(ppi + '\n')
        fw.write('\n')

def samplize_ppi(sid, sent, pair, e_dict):
    p1, p1_type, p1_offset = pair[0]
    p2, p2_type, p2_offset = pair[1]
    ppi = pair[2]
    replaced_sent = ''

    if p1 == p2:
        return None

    if p1_offset.find(',') == -1 and p2_offset.find(',') == -1:
        p1_start, p1_end = p1_offset.split('-')
        p2_start, p2_end = p2_offset.split('-')
        p1_start, p1_end = int(p1_start), int(p1_end)
        p2_start, p2_end = int(p2_start), int(p2_end)
    elif p1_offset.find(',') > -1 and p2_offset.find(',') > -1:
        p1_start, p1_end = p1_offset.split(',')[0].split('-')
        p2_start, p2_end = p2_offset.split(',')[0].split('-')
        p1_start, p1_end = int(p1_start), int(p1_end)
        p2_start, p2_end = int(p2_start), int(p2_end)
    elif p1_offset.find(',') > -1 and p2_offset.find(',') == -1:
        p1_all = p1_offset.split(',')
        p1_all_start, p1_all_end = [], []
        for i in range(len(p1_all)):
            p1_all_start.append(int(p1_all[i].split('-')[0]))
            p1_all_end.append(int(p1_all[i].split('-')[1]))
        # p1_1, p1_2 = p1_offset.split(',')
        # p1_1_start, p1_1_end = p1_1.split('-')
        # p1_2_start, p1_2_end = p1_2.split('-')
        # p1_1_start, p1_1_end = int(p1_1_start), int(p1_1_end)
        # p1_2_start, p1_2_end = int(p1_2_start), int(p1_2_end)
        p2_start, p2_end = p2_offset.split('-')
        p2_start, p2_end = int(p2_start), int(p2_end)

        p1_start, p1_end = p1_all_start[0], p1_all_end[0]
        for i in range(len(p1_all)):
            if len(set(range(p1_all_start[i], p1_all_end[i])) & set(range(p2_start, p2_end))):
                p1_start, p1_end = p1_all_start[i], p1_all_end[i]
                break
    else:
        p2_all = p2_offset.split(',')
        p2_all_start, p2_all_end = [], []
        for i in range(len(p2_all)):
            p2_all_start.append(int(p2_all[i].split('-')[0]))
            p2_all_end.append(int(p2_all[i].split('-')[1]))

        p1_start, p1_end = p1_offset.split('-')
        p1_start, p1_end = int(p1_start), int(p1_end)

        p2_start, p2_end = p2_all_start[0], p2_all_end[0]
        for i in range(len(p2_all)):
            if len(set(range(p2_all_start[i], p2_all_end[i])) & set(range(p1_start, p1_end))):
                p2_start, p2_end = p2_all_start[i], p2_all_end[i]
                break

    # if sid == 'BioInfer.d1052.s1052':
    #     print('1:', replaced_sent)
    # replaced_sent = sent.replace(sent[p1_start:p1_end + 1], 'PROTEINA').replace(sent[p2_start:p2_end + 1], 'PROTEINB')
    if p1_start < p2_start and p1_end < p2_end:
        replaced_sent = sent[: p2_start] + ' PROTEINB ' + sent[p2_end + 1:]
        replaced_sent = replaced_sent[: p1_start] + ' PROTEINA ' + replaced_sent[p1_end + 1:]
    elif p1_start > p2_start and p1_end > p2_end:
        replaced_sent = sent[: p1_start] + ' PROTEINA ' + sent[p1_end + 1:]
        replaced_sent = replaced_sent[: p2_start] + ' PROTEINB ' + replaced_sent[p2_end + 1:]

    # if sid == 'BioInfer.d1052.s1052':
    #     print('2:', replaced_sent)

    other = set(e_dict) - {p1_offset, p2_offset}
    other_new = []
    for n in other:
        n_start, n_end = 0, 0
        if n.find(',') > -1:
            nn = n.split(',')
            for n in nn:
                n_start, n_end = n.split('-')
                n_start, n_end = int(n_start), int(n_end)
                if len(set(range(n_start, n_end)) & set(range(p1_start, p1_end))) <= 0 \
                        and len(set(range(n_start, n_end)) & set(range(p2_start, p2_end))) <= 0:
                    break
        else:
            n_start, n_end = n.split('-')
            n_start, n_end = int(n_start), int(n_end)
        other_new.append([n_start, n_end])

    def sort(ele):
        return ele[0]
    other_new.sort(key=sort, reverse=True)
    # print(other_new)
    replaced_sent = ' ' + preProcess(replaced_sent) + ' '

    replaced_sent = tokenizer.tokenize(replaced_sent)
    other_new = [sent[n[0]:n[1] + 1] for n in other_new]
    replaced_sent = [' PROTEINN ' if x in other_new else x for x in replaced_sent]
    # for n in other_new:
    #     replaced_sent = replaced_sent.replace(sent[n[0]:n[1] + 1], ' PROTEINN ')
    replaced_sent = ' '.join(replaced_sent)

    # if sid == 'BioInfer.d1052.s1052':
    #     # print(sent[n[0]:n[1] + 1])
    #     print('3:', replaced_sent)

    replaced_sent = ' ' + preProcess(replaced_sent) + ' '
    replaced_sent = replaced_sent.replace(" PROTEINA s ", " PROTEINA ")
    replaced_sent = replaced_sent.replace(" PROTEINB s ", " PROTEINB ")
    replaced_sent = replaced_sent.replace(" PROTEINN s ", " PROTEINN ")
    # if sid == 'BioInfer.d1052.s1052':
    #     print('4:', replaced_sent)
    # print(replaced_sent)
    # print(sent)
    if len(replaced_sent.strip()) == 0:
        return None
    
    return [sid, replaced_sent, p1, p1_type, p2, p2_type, ppi]

# def samplize_ddi(sid, sent, pair, e_dict):
#     d1, d1_type, d1_offset = pair[0]
#     d2, d2_type, d2_offset = pair[1]
#     ddi = pair[2]
# 
#     if d1_offset.find(';') == -1 and d2_offset.find(';') == -1:
#         d1_start, d1_end = d1_offset.split('-')
#         d2_start, d2_end = d2_offset.split('-')
#         d1_start, d1_end = int(d1_start), int(d1_end)
#         d2_start, d2_end = int(d2_start), int(d2_end)
#     elif d1_offset.find(';') > -1 and d2_offset.find(';') > -1:
#         d1_start, d1_end = d1_offset.split(';')[0].split('-')
#         d2_start, d2_end = d2_offset.split(';')[0].split('-')
#         d1_start, d1_end = int(d1_start), int(d1_end)
#         d2_start, d2_end = int(d2_start), int(d2_end)
#     elif d1_offset.find(';') > -1 and d2_offset.find(';') == -1:
#         d1_1, d1_2 = d1_offset.split(';')
#         d1_1_start, d1_1_end = d1_1.split('-')
#         d1_2_start, d1_2_end = d1_2.split('-')
#         d1_1_start, d1_1_end = int(d1_1_start), int(d1_1_end)
#         d1_2_start, d1_2_end = int(d1_2_start), int(d1_2_end)
#         d2_start, d2_end = d2_offset.split('-')
#         d2_start, d2_end = int(d2_start), int(d2_end)
# 
#         if len(set(range(d1_1_start, d1_1_end)) & set(range(d2_start, d2_end))):
#             d1_start, d1_end = d1_2_start, d1_2_end
#         else:
#             d1_start, d1_end = d1_1_start, d1_1_end
#     else:
#         d2_1, d2_2 = d2_offset.split(';')
#         d2_1_start, d2_1_end = d2_1.split('-')
#         d2_2_start, d2_2_end = d2_2.split('-')
#         d2_1_start, d2_1_end = int(d2_1_start), int(d2_1_end)
#         d2_2_start, d2_2_end = int(d2_2_start), int(d2_2_end)
#         d1_start, d1_end = d1_offset.split('-')
#         d1_start, d1_end = int(d1_start), int(d1_end)
# 
#         if len(set(range(d2_1_start, d2_1_end)) & set(range(d1_start, d1_end))):
#             d2_start, d2_end = d2_2_start, d2_2_end
#         else:
#             d2_start, d2_end = d2_1_start, d2_1_end
# 
#     other = set(e_dict) - {d1_offset, d2_offset}
#     replaced_sent = sent.replace(sent[d1_start:d1_end + 1], 'DRUGA').replace(sent[d2_start:d2_end + 1], 'DRUGB')
#     for n in other:
#         if n.find(';') > -1:
#             n = n.split(';')[0]
#         n_start, n_end = n.split('-')
#         n_start, n_end = int(n_start), int(n_end)
#         replaced_sent = replaced_sent.replace(sent[n_start:n_end + 1], 'DRUGN')
# 
#     return [sid, replaced_sent, d1, d1_type, d2, d2_type, ddi]


# triplets: [
#           [[sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi],
#            [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi],
#            [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]]
#           ]
def write_triplets_as_txt(triplets, filename):
    fw = open(filename, 'w')
    for triplet in triplets:
        # print(triplet)
        for id, dsent, p1, p1_type, p2, p2_type, ppi in triplet:
            fw.write(id + '\n')
            fw.write(dsent + '\n')
            fw.write(p1 + "\t" + p1_type + "\t" + p2 + "\t" + p2_type + '\n')
            fw.write(ppi + '\n')
            fw.write('\n')
        fw.write('\n\n')


# input: [[sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]]
# output: [ [sent_text], [[e1, type]], [[e2, type]], [label] ]
def read_pickle(filename):
    data = pickle.load(open(filename, 'rb'))
    sent_text = []
    e1_list = []
    e2_list = []
    label = []
    for sent_id, replaced_sent, p1, p1_type, p2, p2_type, ppi in data:

        sent_text.append(replaced_sent.lower())

        # if sent_id == 'DDI-DrugBank.d99.s9':
        #     print(sent_text)
        e1_list.append([p1, p1_type])
        e2_list.append([p2, p2_type])
        label.append(ppi)

    return sent_text, e1_list, e2_list, label