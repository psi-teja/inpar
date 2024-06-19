import os
import json

def clean_pre_label_dict(pre_label_json):
    keys_to_delete = []
    for field, value in pre_label_json.items():
        if isinstance(value, dict):
            if value['location']['pageNo'] == 0:
                keys_to_delete.append(field)
            elif "conf_arr" in value:
                del value['conf_arr']
        elif field in ['Table', 'LedgerDetails']:
            rows_to_delete = []
            for row in value:
                col_to_delete = []
                for col, colValue in row.items():
                    if colValue['location']['pageNo'] == 0:
                        col_to_delete.append(col)
                    elif "conf_arr" in colValue:
                        del colValue['conf_arr']
                for col in col_to_delete:
                    del row[col]
                if not row:
                    rows_to_delete.append(row)
            for row in rows_to_delete:
                value.remove(row)
            
            if not value:
                keys_to_delete.append(field)


        elif isinstance(value, list):
            items_to_delete = []
            for item in value:
                if item['location']['pageNo'] == 0:
                    items_to_delete.append(item)
                elif "conf_arr" in item:
                    del item['conf_arr']
            for item in items_to_delete:
                value.remove(item)

    for key in keys_to_delete:
        del pre_label_json[key]
    
    return pre_label_json

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    json_path = os.path.join(current_dir, "output_pre_label.json")

    with open(json_path, 'r') as f:
        pre_label_dict = json.load(f)

    output = clean_pre_label_dict(pre_label_dict)

    output_path = os.path.join(current_dir, "cleaned_pre_label.json")
    with open(output_path, "w") as json_file:
        json.dump(output, json_file, indent=4)
