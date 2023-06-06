"""
functions for merging the type information from static analysis and machine learning
"""
from preprocess_utils import check, make_types_consistent

def merge_vars(sa_dict, ml_dict):
    add = 0
    var_dict_sa = sa_dict["variables"]
    if "variables_p" in ml_dict.keys():
        var_dict_1 = ml_dict["variables_p"]
    else:
        var_dict_1 = {}
    for var_key in var_dict_sa.keys():
        var_dict_sa[var_key] = make_types_consistent(var_dict_sa[var_key])
        if check(var_dict_sa[var_key]):
            if var_key in var_dict_1.keys():
                if len(var_dict_1[var_key]) != 0 and var_dict_1[var_key][0][0] != var_dict_sa[var_key]:
                        sa_type = [var_dict_sa[var_key], 1.1]
                        var_dict_1[var_key].insert(0, sa_type)
                        add = add + 1
    ml_dict["variables_p"] = var_dict_1
    return ml_dict, add


def merge_params(sa_dict, ml_dict):
    add = 0
    param_dict_sa = sa_dict["params"]
    if "params_p" in ml_dict.keys():
        param_dict_1 = ml_dict["params_p"]
    else:
        param_dict_1 = {}

    for param_key in param_dict_sa.keys():
        if param_key in param_dict_1.keys():
            param_dict_sa[param_key] = make_types_consistent(param_dict_sa[param_key])
            if check(param_dict_sa[param_key]):
                if len(param_dict_1[param_key]) != 0 and param_dict_1[param_key][0][0] != param_dict_sa[param_key]:
                        sa_type = [param_dict_sa[param_key], 1.1]
                        param_dict_1[param_key].insert(0, sa_type)
                        add = add + 1
    ml_dict["params_p"] = param_dict_1
    return ml_dict, add


def merge_ret_types(sa_dict, ml_dict):
    add = 0
    ret_type_sa = sa_dict["ret_type"]
    if "ret_type_p" in ml_dict.keys():
        ret_type_1 = ml_dict["ret_type_p"]
    else:
        ret_type_1 = []
    ret_type_sa = make_types_consistent(ret_type_sa)
    if check(ret_type_sa):
        if len(ret_type_1) != 0 and ret_type_1[0][0] != ret_type_sa:
            sa_type = [ret_type_sa, 1.1]
            ret_type_1.insert(0, sa_type)
            add = add + 1
    ml_dict["ret_type_p"] = ret_type_1
    return ml_dict, add


def merge_file(sa_file, ml_file):
    add_var = 0
    add_params = 0
    add_ret_type = 0

    # merge variables in the py file
    merged_file, add = merge_vars(sa_file, ml_file)
    add_var = add_var + add

    # merge variables, params and ret_types in the functions in the py file
    if "funcs" in ml_file.keys():
        func_list = sa_file['funcs']
        func_list_1 = merged_file['funcs']
        for i in range(len(func_list)):
            if func_list[i]['name'] == func_list_1[i]['name']:
                merged_func, add = merge_vars(func_list[i], func_list_1[i])
                func_list_1[i] = merged_func
                add_var = add_var + add
                merged_func, add = merge_params(func_list[i], func_list_1[i])
                func_list_1[i] = merged_func
                add_params = add_params + add
                merged_func, add = merge_ret_types(func_list[i], func_list_1[i])
                func_list_1[i] = merged_func
                add_ret_type = add_ret_type + add

        merged_file['funcs'] = func_list_1

    if "classes" in ml_file.keys():
        class_list = sa_file['classes']
        class_list_1 = merged_file['classes']
        for i in range(len(class_list)):
            if class_list[i]['name'] == class_list_1[i]['name']:
                # add vars in classes
                class_list_1[i], add = merge_vars(class_list[i], class_list_1[i])
                add_var = add_var + add
                # add vars, params, ret_types in functions in classes
                if "funcs" in class_list_1[i].keys():
                    func_list = class_list[i]['funcs']
                    func_list_1 = class_list_1[i]['funcs']
                    for j in range(len(func_list)):
                        if func_list[j]['name'] == func_list_1[j]['name']:
                            func_list_1[j], add = merge_vars(func_list[j], func_list_1[j])
                            add_var = add_var + add
                            func_list_1[j], add = merge_params(func_list[j], func_list_1[j])
                            add_params = add_params + add
                            func_list_1[j], add = merge_ret_types(func_list[j], func_list_1[j])
                            add_ret_type = add_ret_type + add
                    class_list_1[i]['funcs'] = func_list_1
        merged_file['classes'] = class_list_1

    return merged_file

def merge_project(sa_dict, ml_dict):
    merged_dict = {}
    for key in ml_dict:
        if key in sa_dict.keys():
            m_file = merge_file(sa_dict[key], ml_dict[key])
            merged_dict[key] = m_file
        else:
            merged_dict[key] = ml_dict[key]
    return merged_dict


def update_key(file_name, project_id):
    author = project_id.split("/")[0]
    repo = project_id.split("/")[1]
    list = file_name.split("/")
    start_index = 0
    for i in range(len(list) - 1):
        if list[i] == author and list[i + 1] == repo:
            start_index = i
            break
    new_list = []
    new_list.append("data")
    while start_index < len(list):
        new_list.append(list[start_index])
        start_index = start_index + 1
    return "/".join(new_list)


def merge_pyre(ml_dict, static_dict, project_id):
    src_dict = {}
    src_dict_ml = {}
    for key in static_dict[project_id]['src_files'].keys():
        key_new = update_key(key, project_id)
        src_dict[key_new] = static_dict[project_id]['src_files'][key]
    for key in ml_dict[project_id]['src_files'].keys():
        key_new = update_key(key, project_id)
        src_dict_ml[key_new] = ml_dict[project_id]['src_files'][key]
    merged_project: dict = {project_id: {"src_files": {}}}
    merged_src_dict = merge_project(src_dict, src_dict_ml)
    merged_project[project_id]["src_files"] = merged_src_dict
    return merged_project