import duckdb
from datetime import datetime



def get_metar(dates, filename):
  base = "https://data.source.coop/dynamical/asos-parquet"
  urls = [f"{base}/year={y}/data.parquet" for y in dates]

  df = duckdb.execute("""
      SELECT valid, station, name, tmpf
      FROM read_parquet($1, hive_partitioning=true)
      WHERE station = 'LGA'
      ORDER BY valid
  """, [urls]).fetchdf()

  df.to_csv(filename)

if __name__ == '__main__':
  dates = range(2020, datetime.now().year + 1)
  filename = 'metar_klga.csv'

  get_metar(dates, filename)