# By MW, GPLv3+, may 2021
# Simple script to convert CSV to a searchable table

import datetime, csv

def make_subs(txt, d):
    """Make a series of substitutions"""
    for (k,v) in d.items():
        txt=txt.replace('{'+k+'}',v)
    return txt

def make_table(d):
    t=""
    t+='<tr class="header">'
    for h in d['head']:
        w=10
        if h=="Description":
            w=50
        t+='<th style="width:{}%;">{}</th>'.format(w, h)
    t+='</tr>'
    for i in range(len(d[h])):
        t+='<tr>'
        for j in d['head']:
            t+='<td>{}</td>'.format(d[j][i])
        t+='</tr>'
    return t

if __name__ == "__main__":
    tpl_f = "documentation.tmpl.html"
    htm_f = "documentation.html"
    doc_f = "ChroMag_doc_v10.csv"

    with open(tpl_f, 'r') as f:
        tpl = f.read()

    with open(doc_f, 'r') as f:
        csvf = csv.reader(f)
        for (i,l) in enumerate(csvf):
            if i==0:
                head=l
                csv_d={k:[] for k in head}
                csv_d['head']=head
            else:
                for (k,v) in zip(head,l):
                    csv_d[k].append(v)

    dict_sub = {'date_modified': str(datetime.datetime.now()),
                'table_content': make_table(csv_d),
    }
    txt = make_subs(tpl, dict_sub)

    with open(htm_f, 'w') as f:
        f.write(txt)
    
