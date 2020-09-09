import csv
import os

def get_bone_ages(is_test = False):
    """
        extracts bone's info from the bones csv file and returns two dictionaries bones_ages_dict , bones_is_male_dict
        that given the bone's id return it's age and it's gender
            --test file doesn't contain the ages of the bones, we should consider this.
    """
    if(is_test):
        raise NotImplementedError

    file_name = "boneage-test-dataset.csv" if is_test else "boneage-training-dataset.csv"

    ##### hack ####
    path = os.path.dirname(os.path.realpath(__file__))
    dics = path.split("\\")
    dics[-2] = "torch_scae"
    dics[-1] = "data"
    dics.append("boneage")
    seperator = "\\"
    path = seperator.join(dics)
    ##### hack ####
    
    with open(path + '\\' + file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        bones_ages_dict = {}
        bones_is_male_dict = {}
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                bone_id = int(row[0])
                bone_age = int(row[1])
                is_male = bool(row[2])
                bones_ages_dict[bone_id] = bone_age
                bones_is_male_dict[bone_id] = is_male

    return bones_ages_dict , bones_is_male_dict

if __name__ == "__main__":
    get_bone_ages()