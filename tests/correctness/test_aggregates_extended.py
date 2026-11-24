"""Extended aggregate correctness coverage."""

import pytest

from .datasets import create_aggregate_matrix_pair
from .harness import assert_query_case
from .querygen import aggregate_extended_cases


pytestmark = pytest.mark.extended


AGGREGATE_LAYOUTS = (
    ("segment_by_group", ("group_key",), ("ts", "id")),
    ("segment_by_device", ("device_id",), ("group_key", "ts", "id")),
)

FASTPATH_MODES = (
    ("fastpath_on", "off"),
    ("fastpath_disabled", "on"),
)


@pytest.fixture(params=AGGREGATE_LAYOUTS, ids=lambda layout: layout[0])
def aggregate_matrix(db, request):
    layout_name, segment_by, order_by = request.param
    return create_aggregate_matrix_pair(
        db,
        deltax_table=f"aggregate_extended_{layout_name}",
        segment_by=segment_by,
        order_by=order_by,
    )


@pytest.mark.parametrize("case", list(aggregate_extended_cases()), ids=lambda case: case.name)
@pytest.mark.parametrize("fastpath_mode", FASTPATH_MODES, ids=lambda mode: mode[0])
def test_extended_aggregate_matrix_matches_plain_postgres(
    aggregate_matrix, db, case, fastpath_mode
):
    _, guc_value = fastpath_mode
    plain_table, deltax_table = aggregate_matrix
    db.execute(f"SET pg_deltax.disable_meta_agg_fastpath = {guc_value}")
    assert_query_case(
        db,
        case,
        plain_table=plain_table,
        deltax_table=deltax_table,
    )


# ── FB-0298: DISTINCT em SUM/AVG ────────────────────────────────────────────
#
# O classificador só olhava `is_distinct` no braço de COUNT, então
# `sum(DISTINCT x)` e `avg(DISTINCT x)` viravam Sum/Avg simples no caminho
# DeltaX: a soma de segmento não sabe quais valores repetem e devolvia
# `sum(x)`/`avg(x)` **em silêncio**. Reportado na produção (30/08–03/09:
# 256133 no DeltaX contra 114794 no oráculo, sem precisar de HAVING).
#
# A correção faz o classificador desistir desses dois casos; o teste fixa o
# contrato: o resultado tem de bater com o heap e o dataset tem de ter
# repetição (senão DISTINCT não distingue nada e o teste passaria com o bug).


@pytest.mark.parametrize("agg", ["sum", "avg"])
def test_distinct_aggregate_matches_plain_postgres(aggregate_matrix, db, agg):
    plain_table, deltax_table = aggregate_matrix

    simple_sql = f"SELECT {agg}(device_id) FROM {{table}}"
    distinct_sql = f"SELECT {agg}(DISTINCT device_id) FROM {{table}}"

    plain_distinct = db.execute(distinct_sql.format(table=plain_table)).fetchall()
    plain_simple = db.execute(simple_sql.format(table=plain_table)).fetchall()
    assert plain_distinct != plain_simple, (
        "dataset sem valor repetido em device_id: DISTINCT não distingue nada e o "
        "teste não mediria o FB-0298"
    )

    delta_distinct = db.execute(distinct_sql.format(table=deltax_table)).fetchall()
    assert delta_distinct == plain_distinct, (
        f"{agg}(DISTINCT device_id) divergiu do Postgres puro: deltax={delta_distinct!r} "
        f"plain={plain_distinct!r} (o caminho DeltaX tem de desistir e deixar o heap responder)"
    )
