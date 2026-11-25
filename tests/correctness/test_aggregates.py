"""Aggregate correctness coverage."""

import pytest

from .datasets import create_aggregate_matrix_pair
from .harness import assert_query_case
from .querygen import aggregate_matrix_cases


pytestmark = pytest.mark.smoke


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
        deltax_table=f"aggregate_matrix_{layout_name}",
        segment_by=segment_by,
        order_by=order_by,
    )


@pytest.mark.parametrize("case", list(aggregate_matrix_cases()), ids=lambda case: case.name)
@pytest.mark.parametrize("fastpath_mode", FASTPATH_MODES, ids=lambda mode: mode[0])
def test_aggregate_matrix_matches_plain_postgres(aggregate_matrix, db, case, fastpath_mode):
    _, guc_value = fastpath_mode
    plain_table, deltax_table = aggregate_matrix
    db.execute(f"SET pg_deltax.disable_meta_agg_fastpath = {guc_value}")
    assert_query_case(
        db,
        case,
        plain_table=plain_table,
        deltax_table=deltax_table,
    )


# ── FB-0297: DISTINCT no HAVING ─────────────────────────────────────────────
#
# O casamento HAVING↔SELECT comparava aggfnoid + aggstar + varattno, sem
# `aggdistinct`. Então `count(session_id)` na lista do SELECT era aceito como par
# de `count(DISTINCT session_id)` no HAVING: o grupo era filtrado pela contagem
# ERRADA, em silêncio. Produção (30/08 14:00 → 03/09 21:00, usage_events): 18
# grupos devolvidos contra 1 do oráculo (`valorcode|1934`).
#
# A correção compara o Aggref inteiro (`equal()`), e o teste fixa o contrato: o
# resultado tem de bater com o heap nos dois sentidos (o grupo que passa e o que
# fica de fora), e o dataset tem de ter repetição (senão DISTINCT não distingue).


def test_having_count_distinct_matches_plain_postgres(aggregate_matrix, db):
    plain_table, deltax_table = aggregate_matrix

    sql = """
        SELECT group_key, count(device_id) AS total
          FROM {table}
         GROUP BY group_key
        HAVING count(DISTINCT device_id) > 1
         ORDER BY group_key
    """

    plain_rows = db.execute(sql.format(table=plain_table)).fetchall()
    delta_rows = db.execute(sql.format(table=deltax_table)).fetchall()

    # A query só mede o FB-0297 se DISTINCT mudar o conjunto de grupos.
    sem_distinct = db.execute(
        sql.format(table=plain_table).replace("count(DISTINCT device_id)", "count(device_id)")
    ).fetchall()
    assert plain_rows != sem_distinct, (
        "dataset sem repetição em device_id por group_key: DISTINCT não distingue grupos e o "
        "teste não mediria o FB-0297"
    )

    assert delta_rows == plain_rows, (
        f"HAVING count(DISTINCT device_id) divergiu do Postgres puro: "
        f"deltax={delta_rows!r} plain={plain_rows!r} (o casamento HAVING↔SELECT tem de comparar "
        f"o Aggref inteiro, não só aggfnoid+aggstar+varattno)"
    )
