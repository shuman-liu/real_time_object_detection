import json


def create_index(cat):
    """
    this is the function that will convert the list to a dictionary that the key is the item id
    :param cat:a list contain a series of dictionaries that will link the id to the corresponding category
    :return:a dictionary that will use the id as a key to link the item to id
    """
    index = {}
    for item in cat:
        index[item['id']] = item
    return index


def convert_to_categories(label_map):
    """
    this is the function that will convert the label dictionary obtained from the raw file
    to a list contain a series of dictionaries. these dictionaries will link the id to the corresponding category
    :param label_map:a dictionary that contain the label name, id, categories
    :return:a list contain a series of dictionaries that will link the id to the corresponding category
    """
    cat = []
    for item in label_map:
        if 'display_name' in item:
            name = item['display_name']
        else:
            name = item['name']
        cat.append({'id': item['id'], 'name': name})
    return cat


def load_label(path):
    """
    this is the function that will load the label text file
    to a dictionary that contain the label and corresponding index and label category.
    :param path: the path to the label text file stored
    :return:a dictionary that contain the label name, id, categories
    """
    with open(path) as f:
        lines = f.readlines()
        res = []
        string = ''
        # process the file
        for line in lines:
            line = line.strip('\n')
            if line == "item {":
                string += '{'
            elif line == "}":
                string = string[:-1] + "}"
                res.append(json.loads(string))
                string = ""
            else:
                line = line.split(":")
                string += '"' + line[0].strip(' ') + '":' + line[1] + ","
    return res


if __name__ == "__main__":
    pass
