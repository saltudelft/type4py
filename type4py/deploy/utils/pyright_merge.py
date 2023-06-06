from preprocess_utils import check, make_types_consistent, resolve_type_aliasing

def merge_vars(var_dict_sa, ml_dict, range):
    if "variables_p" in ml_dict.keys():
        var_dict_1 = ml_dict["variables_p"]
    else:
        var_dict_1 = {}
    for var_slot in var_dict_sa:
        var_name = var_slot["name"]
        var_pred = make_types_consistent(var_slot["type"])
        var_pred = resolve_type_aliasing(var_pred)
        # var_dict_sa[var_key] = type_consist(var_dict_sa[var_key])
        if check(var_pred):
            if range == "model":
                if var_name in var_dict_1.keys() and var_slot["loc"] == ml_dict["mod_var_ln"][var_name]:
                    if len(var_dict_1[var_name]) != 0 and var_dict_1[var_name][0][0] != var_pred:
                        sa_type = [var_pred, 1.2]
                        var_dict_1[var_name].insert(0, sa_type)
            if range == "func":
                if var_name in var_dict_1.keys() and var_slot["loc"] == ml_dict["fn_var_ln"][var_name]:
                    if len(var_dict_1[var_name]) != 0 and var_dict_1[var_name][0][0] != var_pred:
                        sa_type = [var_pred, 1.2]
                        var_dict_1[var_name].insert(0, sa_type)
            if range == "class":
                if var_name in var_dict_1.keys() and var_slot["loc"] == ml_dict["cls_var_ln"][var_name]:
                    if len(var_dict_1[var_name]) != 0 and var_dict_1[var_name][0][0] != var_pred:
                        sa_type = [var_pred, 1.2]
                        var_dict_1[var_name].insert(0, sa_type)
    ml_dict["variables_p"] = var_dict_1
    return ml_dict


def merge_params(param_dict_sa, ml_dict):
    if "params_p" in ml_dict.keys():
        param_dict_1 = ml_dict["params_p"]
    else:
        param_dict_1 = {}

    for param_slot in param_dict_sa:
        param_name = param_slot["name"]
        if param_slot["loc"] == ml_dict["fn_lc"] and param_name in param_dict_1.keys():
            type_pred = make_types_consistent(param_slot["type"])
            type_pred = resolve_type_aliasing(type_pred)
            if len(param_dict_1[param_name]) != 0 and param_dict_1[param_name][0][0] != type_pred:
                sa_type = [type_pred, 1.2]
                param_dict_1[param_name].insert(0, sa_type)
    ml_dict["params_p"] = param_dict_1
    return ml_dict


def merge_returns(ret_dict_sa, ml_dict):
    if "ret_type_p" in ml_dict.keys():
        ret_type_1 = ml_dict["ret_type_p"]
    else:
        ret_type_1 = []
    for ret_slot in ret_dict_sa:
        if ret_slot["loc"] == ml_dict["fn_lc"]:
            type_pred = make_types_consistent(ret_slot["type"])
            type_pred = resolve_type_aliasing(type_pred)
            if len(ret_type_1) != 0 and ret_type_1[0][0] != type_pred:
                sa_type = [type_pred, 1.2]
                ret_type_1.insert(0, sa_type)
    ml_dict["ret_type_p"] = ret_type_1
    return ml_dict

def merge_file(sa_file, ml_file):
    filtered_list = [d for d in sa_file if d.get('type') != 'Unknown']
    var_preds = [d for d in filtered_list if d.get("task") == 'var']
    param_preds = [d for d in filtered_list if d.get("task") == "parameter"]
    ret_preds = [d for d in filtered_list if d.get("task") == "return"]

    merged_file = merge_vars(var_preds, ml_file, "model")

    if "funcs" in ml_file.keys():
        func_list = merged_file['funcs']
        func_list_1 = merged_file['funcs']
        for i in range(len(func_list)):
            merged_func = merge_vars(var_preds, func_list[i], "func")
            func_list_1[i] = merged_func
            merged_func = merge_params(param_preds, func_list[i])
            func_list_1[i] = merged_func
            merged_func = merge_returns(ret_preds, func_list[i])
            func_list_1[i] = merged_func

        merged_file['funcs'] = func_list_1

    if "classes" in ml_file.keys():
        class_list = merged_file['classes']
        class_list_1 = merged_file['classes']
        for i in range(len(class_list)):
            class_list_1[i] = merge_vars(var_preds, class_list[i], "class")
            if "funcs" in class_list_1[i].keys():
                func_list = class_list[i]['funcs']
                func_list_1 = class_list_1[i]['funcs']
                for j in range(len(func_list)):
                    merged_func = merge_vars(var_preds, func_list[j], "func")
                    func_list_1[j] = merged_func
                    merged_func = merge_params(param_preds, func_list[j])
                    func_list_1[j] = merged_func
                    merged_func = merge_returns(ret_preds, func_list[j])
                    func_list_1[j] = merged_func

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


def merge_pyright(ml_dict, static_dict, project_id):
    src_dict = {}
    src_dict_ml = {}
    for p_dict in static_dict:
        key = list(p_dict.keys())[0]
        key_new = update_key(key, project_id)
        src_dict[key_new] = p_dict[key]
    for key in ml_dict[project_id]['src_files'].keys():
        key_new = update_key(key, project_id)
        src_dict_ml[key_new] = ml_dict[project_id]['src_files'][key]

    merged_project: dict = {project_id: {"src_files": {}}}
    print(len(src_dict))
    print(len(src_dict_ml))
    merged_src_dict = merge_project(src_dict, src_dict_ml)
    merged_project[project_id]["src_files"] = merged_src_dict
    return merged_project