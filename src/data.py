import duckdb
from src import config as C


def build_sample():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    cases = "\n".join(
        f"    WHEN \"Product\" = '{k}' THEN '{v}'" for k, v in C.PRODUCT_MAP.items()
    )

    con.execute(f"""
    CREATE OR REPLACE VIEW filtered AS
    SELECT
        "Complaint ID"                   AS complaint_id,
        "Date received"                  AS date_received,
        CASE
{cases}
            ELSE NULL
        END                              AS product,
        "Sub-product"                    AS sub_product,
        "Issue"                          AS issue,
        "Consumer complaint narrative"   AS narrative,
        "Company response to consumer"   AS response,
        "Tags"                           AS tags,
        "State"                          AS state
    FROM read_csv_auto('{C.RAW_CSV.as_posix()}', header=true, ignore_errors=true)
    WHERE "Consumer complaint narrative" IS NOT NULL
      AND length("Consumer complaint narrative") > {C.MIN_CHARS}
      AND "Company response to consumer" IS NOT NULL
    """)

    print(con.execute("""
        SELECT product, count(*) AS n
        FROM filtered WHERE product IS NOT NULL
        GROUP BY 1 ORDER BY n DESC
    """).df().to_string())

    con.execute(f"""
    COPY (
        WITH keep AS (
            SELECT product FROM filtered
            WHERE product IS NOT NULL
            GROUP BY 1 HAVING count(*) >= {C.MIN_CLASS_ROWS}
        ),
        deduped AS (
            SELECT * EXCLUDE (dup_rn) FROM (
                SELECT f.*,
                       row_number() OVER (
                           PARTITION BY hash(f.narrative) ORDER BY f.complaint_id
                       ) AS dup_rn
                FROM filtered f
                JOIN keep k USING (product)
            ) WHERE dup_rn = 1
        ),
        ranked AS (
            SELECT d.*,
                   row_number() OVER (
                       PARTITION BY d.product ORDER BY hash(d.complaint_id + {C.SEED})
                   ) AS rn
            FROM deduped d
        )
        SELECT * EXCLUDE (rn),
               CASE WHEN response = 'Closed with monetary relief' THEN 1 ELSE 0 END AS escalated
        FROM ranked WHERE rn <= {C.PER_CLASS}
    ) TO '{C.SAMPLE_PARQUET.as_posix()}' (FORMAT PARQUET);
    """)

    n = con.execute(f"SELECT count(*) FROM '{C.SAMPLE_PARQUET.as_posix()}'").fetchone()[0]
    print(f"\nwrote {n} rows to {C.SAMPLE_PARQUET}")


if __name__ == "__main__":
    build_sample()