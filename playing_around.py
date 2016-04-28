from collections import OrderedDict as odict
old_text_path = 'data/main/DevData/scaleconcept16.teaser_dev_data_textual.scofeat.v20160212'
old_visual_path = 'data/main/DevData/scaleconcept16.teaser_dev_data_visual_vgg16-relu7.dfeat'

new_text_path = 'data/main/DevData/parsed.text'
new_visual_path = 'data/main/DevData/parsed.visual'

imgid_path = 'data/main/DevData/scaleconcept16.teaser_dev.ImgID.txt'
imgToTextId_path = 'data/main/DevData/scaleconcept16.teaser_dev.ImgToTextID.txt'
textit2imgid = odict([])

def makeimgtotext():
    with open(imgToTextId_path) as f:
        for l in f:
            parsed = l.split()
            textit2imgid[parsed[1]] = parsed[0]

def fixTextual():
     with open(new_text_path, 'w') as new_f:
        with open(old_text_path) as old_f:
            for l in old_f:
                splitted = l.split()
                doc_id = splitted[0]
                img_id = textit2imgid[splitted[0]]
                new_line = l.replace(doc_id, img_id, 1)
                new_f.write(new_line)

def fixVisual():
    img_ids = textit2imgid.values()
    with open(new_visual_path, 'w') as new_f:
        with open(old_visual_path) as old_f:
            with open(imgid_path) as img_ids:
                header = old_f.readline()
                new_f.write(header)
                for (img_id,l) in zip(img_ids, old_f):
                    new_f.write(img_id.strip()+' '+ l)


if __name__ =='__main__':
    makeimgtotext()
    # fixTextual()
    fixVisual()