import os
import json
import pandas as pd

# metrics you are interested in
metrics = [
    'wikitext:word_perplexity',
    'lambada_standard:ppl',
    'lambada_standard:acc',
    'piqa:acc',
    'hellaswag:acc_norm',
    'winogrande:acc',
    'arc_easy:acc',
    'arc_challenge:acc_norm',
]

# directory where JSON files are stored
dir_path = 'out/param_eval/'

# a dictionary to collect results
result_data = {}

# walk through folder
for root, dirs, files in os.walk(dir_path):
    files.sort()    
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join(root, file), 'r') as f:
                data = json.load(f)

                # dictionary to store the results for this file
                this_file_data = {}

                for metric in metrics:
                    parts = metric.split(':')
                    if parts[0] in data['results']:
                        this_file_data[metric] = data['results'][parts[0]][
                            parts[1]]

                result_data[file] = this_file_data

# for all metric that contains 'acc', take the avg and add it to last column
# for key, value in result_data.items():
#     result_data[key]['avg_acc'] = sum([
#         value[x] for x in value.keys() if 'acc' in x
#     ]) / len([value[x] for x in value.keys() if 'acc' in x])
# convert the dictionary to a pandas DataFrame and write it to a CSV file
result_df = pd.DataFrame(result_data.values(), index=result_data.keys())
# round to 3
result_df = result_df.round(3)
result_df.to_csv(f'{dir_path}/result_data.csv')
