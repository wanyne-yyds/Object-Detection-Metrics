import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def get_file_content(path_file):
    filepath = Path(path_file).rglob("results.txt")
    PR_dict = defaultdict(list)
    for file in filepath:
        PR_name = osp.split(str(file.parent))[1]
        with open(str(file), "r") as f:
            sumcontent = f.readlines()
            for content in sumcontent:
                linecontent = content.split(':')
                if len(linecontent) == 2 and linecontent[0] in ["Recall"]:
                    PR_Recallname, PR_Recallvalues = linecontent
                    PR_Recallvalues = PR_Recallvalues.strip('\n').strip(' [').strip(']').split(', ')
                    PR_Recallvalues = np.array(list(map(lambda x: float(x.strip('\'')), PR_Recallvalues)))
                    PR_Recall = [PR_Recallname, PR_Recallvalues]

                if len(linecontent) == 2 and linecontent[0] in ["Precision"]:
                    PR_Precisionname, PR_Precisionvalues = linecontent
                    PR_Precisionvalues = PR_Precisionvalues.strip('\n').strip(' [').strip(']').split(', ')
                    PR_Precisionvalues = np.array(list(map(lambda x: float(x.strip('\'')), PR_Precisionvalues)))
                    PR_Precision = [PR_Precisionname, PR_Precisionvalues]

                if len(linecontent) == 2 and linecontent[0] in ["mAP"]:
                    PR_mAP = linecontent

                if len(linecontent) == 2 and linecontent[0] in ["Class"]:
                    PR_classes = linecontent[1].strip('\n')
        PR_dict[PR_name].append([PR_classes, PR_Recall, PR_Precision, PR_mAP])
    return PR_dict                

def draw_pr_curve(pr_values, out_file):
    plt.figure(figsize=(12, 8))
    for keys, values in pr_values.items():
        files_name = keys
        cls, recall, precision, map = values[0]
        recallname, recallvalue = recall
        precisionname, precisionvalue = precision
        mapname, mapvalue = map
        ap_str = " {}: {}".format(mapname, mapvalue)
        plt.plot(recallvalue, precisionvalue, label=files_name + ap_str)
        plt.xlabel(recallname)
        plt.ylabel(precisionname)
        plt.title('Precision x Recall curve')
        plt.legend(shadow=True)
        plt.grid()

    plt.savefig(os.path.join(out_file, 'all.png'))
    plt.close()

if __name__ == "__main__":
    path_file = r'/home/ckn/Code/Object-Detection-Metrics/results/bsd/Two/Big_dataset'
    pr_values = get_file_content(path_file)
    draw_pr_curve(pr_values, path_file)