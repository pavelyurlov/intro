import os
import numpy as np


def clean_labels(labels_folder='../detection_db/labels'):
    for labels_type in ['train', 'val']:
        path = os.path.join(labels_folder, labels_type)
        filenames = os.listdir(path)
        cnt_corrupt = 0
        cnt_good = 0
        for fn in sorted(filenames):
            if fn[-3:] == 'txt':
                filepath = os.path.join(path, fn)
                clipped_labels = np.clip(np.loadtxt(filepath), 0, 1)
                if len(clipped_labels.shape) == 1:
                    cnt_good += 1
                    with open(filepath, 'w') as f:
                        f.write('{} {} {} {} {}'.format(*clipped_labels))
                else:
                    cnt_corrupt += 1
                    os.remove(filepath)
        print('good {} corrupt {} total {}'.format(cnt_good, cnt_corrupt, cnt_good + cnt_corrupt))
        print('{} labels cleaned\n'.format(labels_type))
