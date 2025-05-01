import re
import warnings
from functools import lru_cache
from dataclasses import dataclass
from genlm.control import Potential


class SpiderTableColumnVerifier(Potential):
    def __init__(self, grammar, tables, verbosity=0):
        from lark import Lark

        self.parser = Lark(grammar)
        self.tables = tables
        self.verbosity = verbosity

        super().__init__(list(range(256)))

    @lru_cache(maxsize=None)
    def _parse(self, query):
        return self.parser.parse(query)

    @staticmethod
    def _extract_latest_subquery(query):
        # capture the latest SELECT statement within parentheses (partial or complete)
        subqueries = re.findall(
            r"\(\s*(SELECT[^()]*(?:\([^()]*\)[^()]*)*?)(?:\)|$)",
            query,
            re.IGNORECASE | re.DOTALL,
        )
        return subqueries[-1].strip() if subqueries else None

    @staticmethod
    def _strip_query_at_boundary(query):
        if query.endswith(" WHERE"):
            return query.rstrip(" WHERE")
        elif query.endswith(" GROUP BY"):
            return query.rstrip(" GROUP BY")
        elif query.endswith(" ORDER"):
            return query.rstrip(" ORDER")
        return None

    def _validate(self, parsed):
        validator = ColumnValidator(self.tables, self.verbosity)
        try:
            validator.transform(parsed)
            return 0 if validator.is_valid else float("-inf")
        except Exception:
            return 0

    async def prefix(self, context):
        try:
            context = bytes(context).decode("utf-8")
        except UnicodeDecodeError:
            return 0

        query = self._strip_query_at_boundary(context)
        if not query:
            return 0

        try:
            parsed = self._parse(query)
        except Exception:
            # Try to fallback to the latest subquery.
            try:
                subquery = self._extract_latest_subquery(query)
                if not subquery:
                    return 0

                subquery = self._strip_query_at_boundary(subquery)
                if not subquery:
                    return 0

                parsed = self._parse(subquery)
            except Exception:
                return 0

        return self._validate(parsed)

    async def complete(self, context):
        try:
            context = bytes(context).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")

        try:
            parsed = self._parse(context)
        except Exception:
            warnings.warn(f"Failed to parse complete {context}")
            return float("-inf")

        return self._validate(parsed)


@dataclass
class JoinConstraint:
    expr: str

    def __repr__(self):
        return f"JoinConstraint({self.expr})"  # pragma: no cover

    def __str__(self):
        return f"JoinConstraint({self.expr})"  # pragma: no cover


@dataclass
class Alias:
    alias: str

    def __repr__(self):
        return f"Alias({self.alias})"  # pragma: no cover

    def __str__(self):
        return f"Alias({self.alias})"  # pragma: no cover


try:
    from lark.visitors import Transformer

    HAS_LARK = True
except ImportError:  # pragma: no cover
    HAS_LARK = False  # pragma: no cover

if HAS_LARK:

    class ColumnValidator(Transformer):
        def __init__(self, schema_tables, verbosity=0):
            self.verbosity = verbosity
            self.schema_tables = schema_tables
            self.table_to_columns = {}
            self.is_valid = True

            for t in self.schema_tables:
                self.table_to_columns[t.name.lower()] = [
                    re.sub("_", "", c.name.lower()) for c in t.columns
                ]

        def select_core(self, children):
            (
                _,  # select_,
                _,  # maybe___nt_ws_distinct_or_all,
                _,  # ws,
                column_table_names,  # result_column_plus,
                tables_and_join_constraints,  # maybe___nt_from_table,
                where_columns,  # maybe___nt_ws_where_expr,
                _,  # maybe___nt_group_by,
                _,  # maybe___nt_window_clause
            ) = children

            select_columns = column_table_names.copy()

            table_names = []
            for t in tables_and_join_constraints:
                if isinstance(t, JoinConstraint):
                    column_table_names.extend(t.expr)
                else:
                    table_names.append(t)

            variable2table = {}
            for t in table_names:
                if "." in t:
                    alias = t.split(".")[0].lower()
                    table = t.split(".")[1]
                    variable2table[alias] = table

            for t in self.schema_tables:
                variable2table[t.name.lower()] = t.name.lower()

            if where_columns:
                column_table_names.extend(where_columns)

            if self.verbosity > 1:
                print("variable2table", variable2table)
                print("column_table_names", column_table_names)
                print("table_names", table_names)
                print("tables_and_join_constraints", tables_and_join_constraints)

            for column_table_name in column_table_names:
                if column_table_name == "*":
                    continue

                if "." in column_table_name:  # aliased table
                    alias = column_table_name.split(".")[0].lower()

                    if alias not in variable2table:
                        # undefined table alias
                        if self.verbosity > 0:
                            print(f"Invalid table alias `{alias}`")
                        self.is_valid = False
                        continue

                    table = variable2table[alias]
                    column = column_table_name.split(".")[1]

                    if column not in self.table_to_columns[table]:
                        if self.verbosity > 0:
                            # invalid column
                            print(f"Invalid column name `{column}` for table `{table}`")
                        self.is_valid = False
                else:
                    # check if column is in any of the tables in the from clause
                    # this includes any tables in the join statement
                    valid = False
                    for table in table_names:
                        if "." in table:
                            table = table.split(".")[1].lower()
                        else:
                            table = table.lower()

                        if column_table_name in self.table_to_columns[table]:
                            valid = True
                            break

                    if not valid:
                        if self.verbosity > 0:
                            print(
                                f"Invalid column name `{column_table_name}` for tables `{table_names}`"
                            )
                        self.is_valid = False

            return select_columns

        #####################
        # Column table name #
        #####################

        def result_column_plus(self, children):
            (
                result_column,
                comma_result_column_star,
            ) = children

            return result_column + comma_result_column_star

        def result_column(self, children):
            if len(children) == 2:
                (
                    column_table_name,
                    _,  # maybe___nt_as_column_alias
                ) = children
            elif len(children) == 1:
                column_table_name = children[0]
            else:
                raise ValueError(len(children))
            return column_table_name

        def expr(self, children):
            assert len(children) == 1
            return children[0]  # non_binop_expr

        def non_binop_expr(self, children):
            assert len(children) == 1
            return children[0]  # expr_base_case

        def expr_base_case(self, children):
            assert len(children) == 1
            return children[0]  # possibly_qualified_column_name

        def binop_expr(self, children):
            if len(children) == 1:
                return children[0]  # expr_base_case
            elif len(children) == 3:
                (
                    left,  # non_binop_expr
                    _,  # expr_binop
                    right,  # non_binop_expr
                ) = children
                return left + right
            else:
                raise ValueError(len(children))

        def expr_compound(self, children):
            return sum(children, [])

        def possibly_qualified_column_name(self, children):
            (maybe___nt_qualifier_for_x, column_table) = children

            if maybe___nt_qualifier_for_x:
                return [maybe___nt_qualifier_for_x + "." + column_table]
            else:
                return [column_table]

        def comma_result_column_star(self, children):
            assert len(children) == 1
            return children[0]  # maybe___nt_comma_result_column_plus

        def comma_result_column_plus(self, children):
            (
                _,  # ws_star,
                _,  # comma,
                _,  # ws_star,
                column_table_name,  # result_column
                next_column_table_names,  # comma_result_column_star
            ) = children
            return column_table_name + next_column_table_names

        def aggregate_function_invocation(self, children):
            (
                _,  # aggregate_func
                _,  # ws_star
                _,  # open_paren
                _,  # ws_star
                maybe___nt_distinct_exprs_or_star,
                _,  # ws_star
                _,  # close_par
                _,  # maybe___nt_ws_filter
                _,  # maybe___nt_ws_over
            ) = children
            return maybe___nt_distinct_exprs_or_star

        def distinct_exprs_or_star(self, children):
            if len(children) == 1:
                return ["*"]  # star
            else:
                assert len(children) == 2
                (
                    _,  # maybe___nt_distinct_ws
                    expr_comma_plus,
                ) = children
            return expr_comma_plus

        def expr_comma_plus(self, children):
            (
                expr,
                _,  # comma_expr_star,
            ) = children
            return expr

        ##############
        # Table name #
        ##############

        def from_table(self, children):
            (
                _,  # ws,
                _,  # from_,
                _,  # ws,
                table_name,  # table_or_subquery_plus_or_join
            ) = children
            return table_name

        def table_or_subquery_plus_or_join(self, children):
            assert len(children) == 1
            return children[0]  # join clause

        def join_clause(self, children):
            (
                table_name,  # table_or_subquery
                join_table_names,  # join star
            ) = children
            return join_table_names + [table_name]

        def join_star(self, children):
            assert len(children) == 1
            return children[0]  # maybe___nt_join_plus

        def join_plus(self, children):
            (
                _,  # ws,
                _,  # join_operator,
                table_name,  # table_or_subquery
                join_constraints,  # maybe___nt_ws_join_constraint
                join_star,  # join_star
            ) = children

            table_names = [table_name] + join_star

            if join_constraints:
                table_names.append(JoinConstraint(join_constraints))

            return table_names

        def ws_join_constraint(self, children):
            (
                _,  # ws
                join_constraint,
            ) = children
            return join_constraint

        def join_constraint(self, children):
            (
                _,  # on
                _,  # ws
                expr,
            ) = children
            return expr

        def table_or_subquery(self, children):
            if len(children) == 4:  # table
                (
                    _,  # maybe___nt_schema_dot_ws
                    table_name,  # table_name
                    maybe_alias,  # maybe___nt_as_table
                    _,  # maybe___nt_indexed_or_not
                ) = children
                if maybe_alias:
                    return maybe_alias.alias + "." + table_name
                else:
                    return table_name
            elif len(children) == 2:  # subquery
                (
                    select_columns,  # paren_select
                    maybe_alias,  # maybe___nt_as_table
                ) = children
                table_name = "subquery" + str(hash(str(select_columns)))
                self.table_to_columns[table_name] = select_columns
                if maybe_alias:
                    return maybe_alias.alias + "." + table_name
                else:
                    return table_name
            else:
                raise ValueError(len(children))

        def paren_select(self, children):
            (
                _,  # open_paren
                _,  # ws_star
                select_stmt,  # select_stmt
                _,  # ws_star
                _,  # close_paren
            ) = children
            return select_stmt

        def select_stmt(self, children):
            (
                _,  # maybe___nt_common_table_stmt
                select_core_plus,
                _,  # maybe___nt_ws_order_by
                _,  # maybe___nt_ws_limit
            ) = children
            return select_core_plus

        def select_core_plus(self, children):
            (
                select_core,
                _,  # compound_select_star
            ) = children
            return select_core

        def as_table(self, children):
            (
                _,  # maybe___nt_ws_as,
                _,  # ws,
                table_alias,  # table_alias
            ) = children
            return table_alias

        def table_alias(self, children):
            (
                t,  # t
                n,  # [one, two, three, four, give]
            ) = children
            return Alias(f"t{n}")

        def ws_where_expr(self, children):
            (
                _,  # ws,
                _,  # where_,
                _,  # ws,
                expr,
            ) = children
            return expr

        def table_name(self, children):
            assert len(children) == 1
            return children[0]

        def one(self, children):
            return 1

        def two(self, children):
            return 2

        def three(self, children):
            return 3

        def four(self, children):
            return 4

        def five(self, children):
            return 5

        ########
        # Else #
        ########

        def __default__(self, data, children, meta):
            if data.startswith("maybe___nt_"):
                return [] if len(children) == 0 else children[0]
            elif data.startswith("column_for"):
                return "".join([t.value for t in children])
            elif data.startswith("qualifier_for"):
                (
                    _,  # maybe___nt_schema_dot_ws
                    table_name,
                    _,  # ws_star
                    _,  # dot
                    _,  # ws_star
                ) = children
                return table_name
            elif data.startswith("table_called"):
                if len(children) == 1:
                    assert isinstance(children[0], Alias), children[0]
                    return children[0].alias
                else:
                    return data.value.split("table_called_")[1]
            elif len(data) == 1:
                return data
            else:
                return []

else:

    class ColumnValidator:  # pragma: no cover
        def __init__(self, *args, **kwargs):  # pragma: no cover
            raise ImportError("lark is not installed")  # pragma: no cover
