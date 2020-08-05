import pandas
import os 
from matplotlib import pyplot as plt

ROOT_DIR = "/home/hod/mag/results/finals/"
METRICS = ["loss", "f1_score", "sensitivity", "specificity", "accuracy", "val_loss", "val_f1_score", "val_sensitivity", "val_specificity", "val_accuracy"]
METRICS_LABEL = {
    "val_f1_score": "Validation F1 Score"
}
SPLIT_MAP = {
    "0.2": "80:20",
    "0.3": "70:30",
    "0.4": "60:40"
}
if __name__ == "__main__":
    results = { 
        "EM-CapsNet": {},
        "LeNet-5": {},
         }
    averages = { 
        "EM-CapsNet": {},
        "LeNet-5": {},
    }
    for net in results:
        for split in ["0.4", "0.3", "0.2"]:
            results[net][split] = []

    for (root,dirs,files) in os.walk(ROOT_DIR): 
        for fil in files:
            if fil == 'log.csv':
                split, _, name, _ = root.split("/")[-4:]
                data = pandas.read_csv(f"{root}/log.csv")
                results[name][split].append(data)

    for net in results:
        for split in ["0.4", "0.3", "0.2"]:
            averages[net][split] = {}
            for metric in METRICS:
                sum = 0
                for dp in results[net][split]:
                        sum += dp[metric]
                averages[net][split][metric] = sum/10


    for metric in METRICS:
        title = METRICS_LABEL.get(metric, metric)
        # fig = plt.figure(figsize=(7, 4))
        # # fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
        # for net in averages:
        #     for split in ["0.4", "0.3", "0.2"]:
        #         data = averages[net][split][metric]
        #         plt.plot(range(1, 101), data, label=f"{net}; split {SPLIT_MAP[split]}")
        # plt.legend()
        # plt.title(title)
        # fig.savefig(f"/mnt/mag/results/images/{metric}.png")


        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()
        for net in averages:
            for split in ["0.4", "0.3", "0.2"]:
                data = averages[net][split][metric]
                x = list(range(1, 101))
                x_rev = x[::-1]
                fig.add_trace(go.Scatter(
                    x=x+x_rev,
                    y=data,
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line_color='rgba(255,255,255,0)',
                    showlegend=False,
                    name='Fair',
                ))
                # fig.add_trace(go.Scatter(
                #     x=x, y=y1,
                #     line_color='rgb(0,100,80)',
                #     name='Fair',
                # ))
        # fig.savefig(f"/mnt/mag/results/images/{metric}.png")
        fig.update_traces(mode='lines')
        fig.write_image(f"/mnt/mag/results/images/{metric}.png")
