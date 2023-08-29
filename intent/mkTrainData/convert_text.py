forms = []
with open('./text/order_add_form.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line[0] == '\n':
            continue
        elif line[0] == ';':
            continue
        else:
            forms.append(tuple((line[:-1].split('\t'))))

menus = []
with open('./text/menu.txt', 'r') as f:
    lines = f.readlines()
    menus = [line.split('\n')[0] for line in lines]

index2josa = {
        '0' : ('', ),
        '1' : ('을', '를'),
        '2' : ('이', '가'),
        '3' : ('로', '으로')
        }

orders = []
forms_cmp = []
for menu in menus:
    for josa_idx, text in forms:
        josaList = []
        for idx in josa_idx:
            for josa in index2josa[idx]:
                josaList.append(josa)
        for josa in josaList:
            forms_cmp.append(text.format(menu+josa))

with open('./result/add_order.txt','w') as f:
    for order in forms_cmp:
        f.write(order+'\n')
