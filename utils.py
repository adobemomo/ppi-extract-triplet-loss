

# step1_data: [[sent_id, sent_text, [entity1, entity2, ddi]]]
def write_step1_data_as_txt(data, filename):
    fw = open(filename, 'w')
    for sid, stext, pair in data:
        if len(pair) == 0:
            continue
        fw.write(sid + "\t" + stext + "\n")
        for e1, e2, ddi in pair:
            fw.write(e1[0] + '\t' + e1[1] + '\t' + e1[2] + '\t' + e2[0] + '\t' + e2[1] + '\t' + e2[2] + '\t' + ddi)
            fw.write('\n')
        fw.write('\n')

# step2_data: [[sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]]
def write_step2_data_as_txt(data, filename):
    fw = open(filename, 'w')
    for id, dsent, d1, d1_type, d2, d2_type, ddi in data:
        fw.write(id + '\n')
        fw.write(dsent + '\n')
        fw.write(d1 + "\t" + d1_type + "\t" + d2 + "\t" + d2_type + '\n')
        fw.write(ddi + '\n')
        fw.write('\n')


def samplize(sid, sent, pair, e_dict):
    d1, d1_type, d1_offset = pair[0]
    d2, d2_type, d2_offset = pair[1]
    ddi = pair[2]

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

    other = set(e_dict) - {d1_offset, d2_offset}
    replaced_sent = sent.replace(sent[d1_start:d1_end + 1], 'DRUGA').replace(sent[d2_start:d2_end + 1], 'DRUGB')
    for n in other:
        if n.find(';') > -1:
            n = n.split(';')[0]
        n_start, n_end = n.split('-')
        n_start, n_end = int(n_start), int(n_end)
        replaced_sent = replaced_sent.replace(sent[n_start:n_end + 1], 'DRUGN')

    return [sid, replaced_sent, d1, d1_type, d2, d2_type, ddi]


# triplets: [
#           [[sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi],
#            [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi],
#            [sent_id, replaced_sent, d1, d1_type, d2, d2_type, ddi]]
#           ]
def write_triplets_as_txt(triplets, filename):
    fw = open(filename, 'w')
    for triplet in triplets:
        # print(triplet)
        for id, dsent, d1, d1_type, d2, d2_type, ddi in triplet:
            fw.write(id + '\n')
            fw.write(dsent + '\n')
            fw.write(d1 + "\t" + d1_type + "\t" + d2 + "\t" + d2_type + '\n')
            fw.write(ddi + '\n')
            fw.write('\n')
        fw.write('\n\n')