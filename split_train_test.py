from definitions import Sentence_props
import pickle
with open('temprel_data', 'rb') as fp:
    all_data = pickle.load(fp)

all_rel = {}
for elem in all_data:
    try:
        all_rel[elem.relation] += 1
    except KeyError:
        all_rel[elem.relation] = 1
print all_rel

for key in all_rel:
    all_rel[key] = int(0.8*all_rel[key])

test_file = []
train_file = []
for elem in all_data:
    if all_rel[elem.relation] > 0:
        train_file.append(elem)
        all_rel[elem.relation] -= 1
    else:
        test_file.append(elem)

with open('temprel_data_train', 'wb') as fp:
    pickle.dump(train_file, fp)

with open('temprel_data_test', 'wb') as fp:
    pickle.dump(test_file, fp)

print len(train_file), len(test_file)

