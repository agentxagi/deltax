"""Ida e volta por tipo: comprimir não pode mudar o que a consulta devolve.

POR QUE ESTE ARQUIVO EXISTE

Em 2026-07-26 um ledger de faturamento com seis semanas de dados ficou ilegível em
produção sem ninguém notar. `numeric` lido de partição comprimida devolvia até
73 mil dígitos, `SUM()` estourava com `numeric field overflow`, e `WHERE` sobre
outra coluna trazia linhas erradas. O dado estava intacto: o caminho de leitura
entregava o *texto* armazenado embrulhado como se fosse um Datum daquele tipo, e o
`numeric_out` do Postgres lia aquilo como dígitos base-10000.

O bug passou porque a suíte cobria os tipos que o compressor especializa (int,
float, bool, timestamp, text, jsonb) e nenhum dos que caem no caminho genérico.

A regra que este arquivo impõe é uma só: **para todo tipo suportado, a mesma
consulta tem de devolver o mesmo resultado antes e depois de comprimir.** Vale
para o valor, para o filtro e para a agregação — os três lugares onde o tipo é
interpretado.
"""

import pytest

pytestmark = pytest.mark.smoke


# (nome da coluna, tipo SQL, expressão que gera o valor a partir de g)
#
# Cada linha é um tipo que a extensão declara suportar. Acrescentar tipo aqui é o
# jeito de declarar suporte: se ele não sobreviver ao ciclo, o teste falha.
TYPE_CASES = [
    ("c_int2", "smallint", "(g % 300)::smallint"),
    ("c_int4", "integer", "(g * 7)::integer"),
    ("c_int8", "bigint", "(g::bigint * 1000000007)"),
    ("c_float4", "real", "(g::real / 3.0)"),
    ("c_float8", "double precision", "(g::float8 / 7.0)"),
    ("c_numeric", "numeric(18,8)", "(-0.002 * (1 + (g % 7)))::numeric(18,8)"),
    ("c_numeric_big", "numeric", "(g::numeric * 123456789.987654321)"),
    ("c_bool", "boolean", "(g % 2 = 0)"),
    ("c_text", "text", "'texto-' || g"),
    ("c_varchar", "varchar(32)", "('vc-' || g)::varchar(32)"),
    ("c_uuid", "uuid", "('00000000-0000-4000-8000-' || lpad(g::text, 12, '0'))::uuid"),
    ("c_date", "date", "('2026-01-01'::date + (g % 365))"),
    ("c_ts", "timestamp", "('2026-01-01 00:00:00'::timestamp + (g || ' minutes')::interval)"),
    ("c_jsonb", "jsonb", """jsonb_build_object('n', g, 's', 'v' || g)"""),
]

ROWS = 5000


def _closed_partition_start(cur, table):
    """Início da partição fechada mais recente.

    O compressor recusa partição cujo `range_end` ainda está no futuro, então o
    teste precisa gravar dentro de uma faixa já encerrada — é também o cenário de
    produção.
    """
    cur.execute(
        """
        SELECT range_start, partition_name
          FROM deltax.deltax_partition_info(%s)
         WHERE range_end < now()
         ORDER BY range_end DESC
         LIMIT 1
        """,
        (table,),
    )
    row = cur.fetchone()
    assert row is not None, "nenhuma partição fechada; ajuste o intervalo do teste"
    return row[0], row[1]


@pytest.fixture()
def roundtrip_table(db):
    """Tabela deltax com uma coluna de cada tipo, populada numa partição fechada."""
    cols = ",\n  ".join(f'"{name}" {sql_type}' for name, sql_type, _ in TYPE_CASES)
    exprs = ",\n  ".join(expr for _, _, expr in TYPE_CASES)
    names = ", ".join(f'"{name}"' for name, _, _ in TYPE_CASES)

    with db.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS roundtrip CASCADE")
        cur.execute(
            "DELETE FROM deltax.deltax_deltatable WHERE table_name = 'roundtrip'"
        )
        cur.execute(
            f"""
            CREATE TABLE roundtrip (
              tenant_id text NOT NULL,
              reason    text NOT NULL,
              {cols},
              created_at timestamptz NOT NULL
            ) PARTITION BY RANGE (created_at)
            """
        )
        cur.execute(
            "SELECT deltax.deltax_create_table('public.roundtrip', 'created_at', '30 days')"
        )
        cur.execute(
            """
            SELECT deltax.deltax_enable_compression(
              'public.roundtrip',
              ARRAY['tenant_id', 'reason'],
              ARRAY['tenant_id', 'created_at'],
              30000
            )
            """
        )
        start, partition = _closed_partition_start(cur, "public.roundtrip")
        # Sem parâmetros na INSERT: as expressões usam `%` como módulo, e o
        # psycopg trataria `%` como placeholder se `params` fosse passado.
        cur.execute(
            f"""
            INSERT INTO roundtrip (tenant_id, reason, {names}, created_at)
            SELECT (ARRAY['tenant-a','tenant-b'])[1 + (g % 2)],
                   (ARRAY['embedding','mcp_call'])[1 + (g % 2)],
                   {exprs},
                   '{start.isoformat()}'::timestamptz + (g || ' seconds')::interval
              FROM generate_series(1, {ROWS}) g
            """
        )
        db.commit()
        yield partition
        cur.execute("DROP TABLE IF EXISTS roundtrip CASCADE")
        cur.execute(
            "DELETE FROM deltax.deltax_deltatable WHERE table_name = 'roundtrip'"
        )
        db.commit()


def _snapshot(cur, sql):
    cur.execute(sql)
    return cur.fetchall()


def _compress(cur, partition):
    cur.execute("SELECT deltax.deltax_compress_partition(%s)", (partition,))
    return cur.fetchone()[0]


def test_every_value_survives_compression(db, roundtrip_table):
    """Linha por linha, coluna por coluna: o valor lido é o mesmo."""
    names = ", ".join(f'"{name}"' for name, _, _ in TYPE_CASES)
    sql = f"SELECT {names} FROM roundtrip ORDER BY created_at"

    with db.cursor() as cur:
        before = _snapshot(cur, sql)
        assert len(before) == ROWS

        _compress(cur, roundtrip_table)
        db.commit()

        after = _snapshot(cur, sql)

    assert len(after) == len(before), "comprimir mudou a contagem de linhas"

    # Compara por coluna para a falha dizer QUAL tipo quebrou, em vez de
    # despejar 5 mil tuplas.
    for col_idx, (name, sql_type, _) in enumerate(TYPE_CASES):
        col_before = [r[col_idx] for r in before]
        col_after = [r[col_idx] for r in after]
        if col_before != col_after:
            first = next(
                i for i, (a, b) in enumerate(zip(col_before, col_after)) if a != b
            )
            pytest.fail(
                f"{name} ({sql_type}) não sobreviveu à compressão. "
                f"Primeira divergência na linha {first}: "
                f"antes={col_before[first]!r} depois={col_after[first]!r}"
            )


def test_aggregates_survive_compression(db, roundtrip_table):
    """SUM, MIN, MAX e COUNT dão o mesmo número.

    É onde o bug de produção apareceu primeiro: `SUM(amount)` estourava com
    `numeric field overflow` porque cada valor voltava com dezenas de milhares de
    dígitos.
    """
    sql = """
        SELECT count(*),
               sum(c_int4), min(c_int4), max(c_int4),
               sum(c_int8),
               round(sum(c_numeric), 8)::text,
               min(c_numeric)::text, max(c_numeric)::text,
               round(sum(c_numeric_big), 6)::text,
               min(c_text), max(c_text),
               min(c_uuid::text), max(c_uuid::text),
               min(c_date)::text, max(c_date)::text,
               count(*) FILTER (WHERE c_bool)
          FROM roundtrip
    """
    with db.cursor() as cur:
        before = _snapshot(cur, sql)
        _compress(cur, roundtrip_table)
        db.commit()
        after = _snapshot(cur, sql)

    assert before == after, f"agregação divergiu:\nantes={before}\ndepois={after}"


@pytest.mark.parametrize(
    "predicate",
    [
        "c_numeric < -0.005",
        "c_numeric = -0.002",
        "c_numeric_big > 1000000000",
        "c_int4 BETWEEN 100 AND 200",
        "c_int8 > 3000000021",
        "c_text = 'texto-42'",
        "c_text LIKE 'texto-1%'",
        "c_varchar = 'vc-99'",
        "c_uuid = '00000000-0000-4000-8000-000000000042'::uuid",
        "c_bool",
        "c_date > '2026-06-01'",
        "c_float8 > 100.0",
        "c_jsonb->>'s' = 'v7'",
    ],
)
def test_filters_survive_compression(db, roundtrip_table, predicate):
    """O mesmo WHERE seleciona as mesmas linhas.

    Valor correto e filtro correto são problemas distintos: um tipo cujo payload
    comprimido é o texto pode comparar lexicograficamente ("10" < "9") e devolver
    o conjunto errado mesmo quando cada valor lido está certo.
    """
    sql = f"SELECT count(*), min(created_at)::text FROM roundtrip WHERE {predicate}"
    with db.cursor() as cur:
        before = _snapshot(cur, sql)
        _compress(cur, roundtrip_table)
        db.commit()
        after = _snapshot(cur, sql)

    assert before == after, (
        f"WHERE {predicate} mudou depois de comprimir: antes={before} depois={after}"
    )


def test_decompress_restores_exactly(db, roundtrip_table):
    """Descomprimir devolve o que entrou, byte a byte no que dá para comparar."""
    names = ", ".join(f'"{name}"' for name, _, _ in TYPE_CASES)
    sql = f"SELECT {names} FROM roundtrip ORDER BY created_at"

    with db.cursor() as cur:
        before = _snapshot(cur, sql)
        _compress(cur, roundtrip_table)
        db.commit()
        cur.execute(
            "SELECT deltax.deltax_decompress_partition(%s)", (roundtrip_table,)
        )
        db.commit()
        after = _snapshot(cur, sql)

    assert after == before, "descomprimir não devolveu os valores originais"


def test_unsupported_type_is_refused_not_corrupted(db):
    """Tipo que a extensão não sabe reconstruir é recusado na hora de comprimir.

    Antes, um tipo desconhecido caía no caminho de texto e voltava como lixo. Falhar
    alto na compressão é a única alternativa aceitável: o cliente descobre na hora,
    não seis semanas depois, num relatório que não fecha.
    """
    with db.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS unsupported_probe CASCADE")
        cur.execute(
            "DELETE FROM deltax.deltax_deltatable WHERE table_name = 'unsupported_probe'"
        )
        cur.execute(
            """
            CREATE TABLE unsupported_probe (
              tenant_id  text NOT NULL,
              weird      int4range,
              created_at timestamptz NOT NULL
            ) PARTITION BY RANGE (created_at)
            """
        )
        cur.execute(
            "SELECT deltax.deltax_create_table('public.unsupported_probe', 'created_at', '30 days')"
        )
        cur.execute(
            """
            SELECT deltax.deltax_enable_compression(
              'public.unsupported_probe', ARRAY['tenant_id'], ARRAY['created_at'], 30000
            )
            """
        )
        start, partition = _closed_partition_start(cur, "public.unsupported_probe")
        cur.execute(
            """
            INSERT INTO unsupported_probe (tenant_id, weird, created_at)
            SELECT 'tenant-a', int4range(g, g + 10), %s::timestamptz + (g || ' seconds')::interval
              FROM generate_series(1, 100) g
            """,
            (start,),
        )
        db.commit()

        with pytest.raises(Exception) as excinfo:
            cur.execute("SELECT deltax.deltax_compress_partition(%s)", (partition,))
        db.rollback()
        msg = str(excinfo.value).lower()
        assert "int4range" in msg or "unsupported" in msg or "não suportado" in msg, (
            f"a recusa precisa nomear a coluna e o tipo; veio: {excinfo.value}"
        )

        cur.execute("DROP TABLE IF EXISTS unsupported_probe CASCADE")
        cur.execute(
            "DELETE FROM deltax.deltax_deltatable WHERE table_name = 'unsupported_probe'"
        )
        db.commit()
