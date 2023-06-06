import libcst as cst
from libcst.metadata import PositionProvider, CodeRange


class TypeAnnotationFinder(cst.CSTTransformer):
    """
    It find all type annotations from a source code
    use a dict object for saving the type information:
    for example:
        {
            dt: "param",
            func_name: "my_foo",
            name: "foo",
            label: "Fool"
        }
    """
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self):
        super().__init__()
        self.annotated_types = []
        self.var_list = set()

    def __get_line_column_no(self, node):
        lc = self.get_metadata(cst.metadata.PositionProvider, node)
        return (lc.start.line, lc.start.column), (lc.end.line, lc.end.column)

    def visit_FunctionDef(self, original_node: cst.FunctionDef):
        fn_pos = self.__get_line_column_no(original_node)
        if original_node.returns is not None:
            node_module = cst.Module([original_node.returns.annotation])
            type_dict = dict(dt="ret", func_name=original_node.name.value, name="ret_type",
                             label=node_module.code, loc = fn_pos)
            self.annotated_types.append(type_dict)
        for param in original_node.params.params:
            if param.annotation is not None:
                node_module = cst.Module([param.annotation.annotation])
                type_dict = dict(dt="param", func_name=original_node.name.value, name=param.name.value,
                                 label=node_module.code, loc=fn_pos)
                self.annotated_types.append(type_dict)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        pos = self.__get_line_column_no(node.target)
        node_module = cst.Module([node.annotation.annotation])
        var_name = cst.Module([node.target]).code
        if pos not in self.var_list:
            type_dict = dict(dt="var", func_name="__global__", name=var_name,
                             label=node_module.code, loc=pos)
            self.annotated_types.append(type_dict)
            self.var_list.add(pos)


class TypeAnnotationMasker(cst.CSTTransformer):
    """
    It removes type annotations and add reveal_type() for static analysis
    """
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, tar_dict):
        super().__init__()
        self.target = tar_dict
        self.dt = tar_dict["dt"]
        self.find = False

    def __get_line_column_no(self, node):
        lc = self.get_metadata(cst.metadata.PositionProvider, node)
        return (lc.start.line, lc.start.column), (lc.end.line, lc.end.column)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node):
        fn_pos = self.__get_line_column_no(original_node)
        if self.find == False and fn_pos == self.target["loc"]:

            # for return types
            if self.dt == "ret" and original_node.returns is not None:
                log_stmt = cst.Expr(cst.parse_expression(f"reveal_type({updated_node.name.value})"))
                self.find = True
                return cst.FlattenSentinel([updated_node.with_changes(returns=None), log_stmt])

            # for parameterss
            elif self.dt == "param":
                updated_params = []
                for param in original_node.params.params:
                    if param.name.value == self.target["name"]:
                        self.find = True
                        param_untyped = param.with_changes(annotation=None, comma=None)
                        updated_params.append(param_untyped)
                    else:
                        updated_params.append(param)
                log_stmt = cst.Expr(cst.parse_expression(f"reveal_type({updated_node.name.value})"))
                return cst.FlattenSentinel([updated_node.with_changes(
                    params=cst.Parameters(updated_params)), log_stmt]
                )

            else:
                return updated_node

        else:
            return updated_node

    def leave_AnnAssign(self, original_node: cst.AnnAssign, updated_node) -> None:
        if self.find == False and self.dt == "var" and self.target["func_name"] == "__global__":
            pos = self.__get_line_column_no(original_node.target)
            var_name = cst.Module([original_node.target]).code
            if pos == self.target["loc"] and var_name == self.target["name"]:
                self.find = True
                log_stmt = cst.Expr(cst.parse_expression(f"reveal_type({updated_node.target.value})"))
                if original_node.value == None:
                    updated_node = cst.Assign(targets=[cst.AssignTarget(target=original_node.target)],
                                              value=cst.Ellipsis())
                else:
                    updated_node = cst.Assign(targets=[cst.AssignTarget(target=original_node.target)],
                                              value=original_node.value)
                return cst.FlattenSentinel([updated_node, log_stmt])
            else:
                return updated_node
        else:
            return updated_node
