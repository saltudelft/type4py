
def extract_var(label_unit, process_unit):
    if "variables" in label_unit.keys() and "variables_p" in process_unit.keys():
        var_l = label_unit["variables"]
        var_p_l = process_unit["variables_p"]
        var_list_unit = []
        if len(var_l)!=0:
            for var_name in var_l.keys():
                label = var_l[var_name]
                if var_name in var_p_l.keys():
                    predicts = var_p_l[var_name]
                    var_list_unit.append({"original_type" : label, "t4py_predicts": predicts, "task": "variable"})
        return var_list_unit
    else:
        return None

def extract_param(label_unit, process_unit):
    if "params" in label_unit.keys() and "params_p" in process_unit.keys():
        param_l = label_unit["params"]
        param_p_l = process_unit["params_p"]
        param_list_unit = []
        if len(param_l)!=0:
            for param_name in param_l.keys():
                label = param_l[param_name]
                if param_name in param_p_l.keys():
                    predicts = param_p_l[param_name]
                    param_list_unit.append({"original_type" : label, "t4py_predicts": predicts, "task": "parameter"})
        return param_list_unit
    else:
        return None

def extract_ret(label_unit, process_unit):
    if "ret_type" in label_unit.keys() and "ret_type_p" in process_unit.keys():
        ret_unit = []
        ret_l = label_unit["ret_type"]
        ret_p = process_unit["ret_type_p"]
        ret_unit.append({"original_type": ret_l, "t4py_predicts": ret_p, "task": "return types"})
        return ret_unit
    else:
        return None



def extract_file(label_f, process_f):
    type_f = []
    var_list_f = []
    param_list_f = []
    ret_list_f = []

    var_list = extract_var(label_f, process_f)
    if var_list is not None and len(var_list)!=0:
        var_list_f.extend(var_list)

    if "funcs" in label_f.keys() and "funcs" in process_f.keys():
        for i in range(len(label_f["funcs"])):
            label_func = label_f["funcs"][i]
            process_func = process_f["funcs"][i]
            if label_func['name'] == process_func['name']:
                var_list = extract_var(label_func, process_func)
                if var_list is not None and len(var_list) != 0:
                    var_list_f.extend(var_list)
                param_list = extract_param(label_func, process_func)
                if param_list is not None and len(param_list) != 0:
                    param_list_f.extend(param_list)
                ret_list = extract_ret(label_func, process_func)
                if ret_list is not None and len(ret_list) != 0:
                    ret_list_f.extend(ret_list)

    if "classes" in label_f.keys() and "classes" in process_f.keys():
        for j in range(len(label_f["classes"])):
            label_class = label_f['classes'][j]
            process_class = process_f['classes'][j]
            if label_class["name"] == process_class["name"]:
                var_list = extract_var(label_class, process_class)
                if var_list is not None and len(var_list) != 0:
                    var_list_f.extend(var_list)
                if "funcs" in label_class.keys() and "funcs" in process_class.keys():
                    for k in range(len(label_class["funcs"])):
                        label_func = label_class["funcs"][k]
                        process_func = process_class["funcs"][k]
                        if label_func['name'] == process_func['name']:
                            var_list = extract_var(label_func, process_func)
                            if var_list is not None and len(var_list) != 0:
                                var_list_f.extend(var_list)
                            param_list = extract_param(label_func, process_func)
                            if param_list is not None and len(param_list) != 0:
                                param_list_f.extend(param_list)
                            ret_list = extract_ret(label_func, process_func)
                            if ret_list is not None and len(ret_list) != 0:
                                ret_list_f.extend(ret_list)

    type_f.extend(var_list_f)
    type_f.extend(param_list_f)
    type_f.extend(ret_list_f)
    return type_f


def extract_result_ml(label_file, processed_file, project_id):
    type_project = []
    for key in processed_file[project_id]['src_files'].keys():
        typelist_f = extract_file(label_file[project_id]['src_files'][key], processed_file[project_id]['src_files'][key])
        type_project.extend(typelist_f)
    return type_project

