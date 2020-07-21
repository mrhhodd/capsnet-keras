import os

BASE_DIR = "/net/people/plgmwnetrzak/magisterka/result"

def analyze_log_file(fil):
    with open(fil) as f:
        result = {}
        content = f.readlines()
        for line in content[1:]:
            for val, name in zip(line.split(','), content[0].split(',')):
                if 'val_' in name:
                    new_name = name.replace('\n','').replace('val_','')
                    if not result.get(new_name):
                        result[new_name] = []
                    result[new_name].append(float(val.replace('\n','')))
        for param in result:
            result[param] = str(max(result[param])).replace('.', ',')
        return result

if __name__ == "__main__":
    result = []
    for (root,dirs,files) in os.walk(BASE_DIR): 
        for fil in files:
            if fil == 'log.csv':
                log_file = f"{root}/{fil}"
                new_item = {'name': root}
                new_item.update(analyze_log_file(log_file))
                result.append(new_item)
    with open('result.csv', 'w') as f:
        for item in result:
            f.write(f"{item['name']} {item.get('f1_score', '0')}\n")