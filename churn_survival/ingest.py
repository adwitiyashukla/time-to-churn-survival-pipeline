from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TRANSACTION_DTYPES = {
    "msno": "string",
    "payment_method_id": "int16",
    "payment_plan_days": "int32",
    "plan_list_price": "int32",
    "actual_amount_paid": "int32",
    "is_auto_renew": "int8",
    "transaction_date": "int32",
    "membership_expire_date": "int32",
    "is_cancel": "int8",
}

MEMBER_DTYPES = {
    "msno": "string",
    "city": "int16",
    "bd": "int32",
    "gender": "string",
    "registered_via": "int16",
    "registration_init_time": "int32",
}

GENDER_CODES = {"": 0, "male": 1, "female": 2}

TRANSACTION_SCHEMA = pa.schema(
    [
        ("user_id", pa.int32()),
        ("payment_method_id", pa.int16()),
        ("payment_plan_days", pa.int16()),
        ("plan_list_price", pa.int32()),
        ("actual_amount_paid", pa.int32()),
        ("is_auto_renew", pa.int8()),
        ("is_cancel", pa.int8()),
        ("txn_day", pa.int32()),
        ("expire_day", pa.int32()),
    ]
)

MIN_DAY = -25567
MAX_DAY = 29220


def yyyymmdd_to_day(values):
    raw = np.asarray(values, dtype=np.int64)
    years = raw // 10000
    months = (raw // 100) % 100
    days = raw % 100
    valid = (
        (years >= 1990)
        & (years <= 2050)
        & (months >= 1)
        & (months <= 12)
        & (days >= 1)
        & (days <= 31)
    )
    months_since_epoch = np.where(valid, (years - 1970) * 12 + (months - 1), 0)
    month_start = months_since_epoch.astype("datetime64[M]").astype("datetime64[D]")
    result = month_start + np.where(valid, days - 1, 0).astype("timedelta64[D]")
    day_index = result.astype("datetime64[D]").astype(np.int64)
    month_length = (
        (months_since_epoch + 1).astype("datetime64[M]").astype("datetime64[D]")
        - month_start
    ).astype(np.int64)
    valid = valid & (days <= month_length)
    return np.where(valid, day_index, np.iinfo(np.int32).min).astype(np.int64), valid


def day_to_yyyymmdd(days):
    stamps = np.asarray(days, dtype="int64").astype("datetime64[D]")
    years = stamps.astype("datetime64[Y]").astype(np.int64) + 1970
    months = stamps.astype("datetime64[M]").astype(np.int64) % 12 + 1
    day_of_month = (stamps - stamps.astype("datetime64[M]")).astype(np.int64) + 1
    return (years * 10000 + months * 100 + day_of_month).astype(np.int64)


class UserIndex:
    def __init__(self):
        self._index = pd.Index([], dtype="object")

    def assign(self, values):
        codes = self._index.get_indexer(values)
        missing = codes < 0
        if missing.any():
            fresh = pd.Index(pd.unique(values[missing]))
            self._index = self._index.append(fresh)
            codes = self._index.get_indexer(values)
        return codes.astype(np.int64)

    def frame(self):
        return pd.DataFrame(
            {
                "user_id": np.arange(len(self._index), dtype=np.int32),
                "msno": self._index.to_numpy(),
            }
        )

    def __len__(self):
        return len(self._index)


def ingest_transactions(config, verbose=True):
    source = config.paths.raw_dir / config.data.transactions_file
    if not source.is_file():
        raise FileNotFoundError("no transactions file at %s" % source)
    config.paths.interim_dir.mkdir(parents=True, exist_ok=True)
    target = config.paths.interim_dir / "transactions.parquet"
    users_target = config.paths.interim_dir / "users.parquet"

    start_day = int(np.datetime64(str(config.data.observation_start)).astype("datetime64[D]").astype(int))
    end_day = int(np.datetime64(str(config.data.observation_end)).astype("datetime64[D]").astype(int))
    max_users = config.data.max_users

    index = UserIndex()
    writer = pq.ParquetWriter(target, TRANSACTION_SCHEMA, compression="zstd")
    kept = 0
    seen = 0
    dropped_dates = 0
    dropped_window = 0

    reader = pd.read_csv(
        source,
        dtype=TRANSACTION_DTYPES,
        chunksize=config.data.chunk_rows,
        usecols=list(TRANSACTION_DTYPES),
    )
    try:
        for position, chunk in enumerate(reader):
            seen += len(chunk)
            txn_day, txn_ok = yyyymmdd_to_day(chunk["transaction_date"].to_numpy())
            expire_day, expire_ok = yyyymmdd_to_day(chunk["membership_expire_date"].to_numpy())
            good = txn_ok & expire_ok
            dropped_dates += int((~good).sum())
            in_window = good & (txn_day >= start_day) & (txn_day <= end_day)
            dropped_window += int((good & ~in_window).sum())
            chunk = chunk.loc[in_window]
            txn_day = txn_day[in_window]
            expire_day = expire_day[in_window]
            if chunk.empty:
                continue
            user_id = index.assign(chunk["msno"].to_numpy(dtype=object))
            if max_users > 0:
                keep = user_id < max_users
                if not keep.any():
                    continue
                chunk = chunk.loc[keep]
                user_id = user_id[keep]
                txn_day = txn_day[keep]
                expire_day = expire_day[keep]
            table = pa.table(
                {
                    "user_id": pa.array(user_id.astype(np.int32)),
                    "payment_method_id": pa.array(chunk["payment_method_id"].to_numpy()),
                    "payment_plan_days": pa.array(
                        np.clip(chunk["payment_plan_days"].to_numpy(), -32768, 32767).astype(
                            np.int16
                        )
                    ),
                    "plan_list_price": pa.array(chunk["plan_list_price"].to_numpy()),
                    "actual_amount_paid": pa.array(chunk["actual_amount_paid"].to_numpy()),
                    "is_auto_renew": pa.array(chunk["is_auto_renew"].to_numpy()),
                    "is_cancel": pa.array(chunk["is_cancel"].to_numpy()),
                    "txn_day": pa.array(txn_day.astype(np.int32)),
                    "expire_day": pa.array(expire_day.astype(np.int32)),
                },
                schema=TRANSACTION_SCHEMA,
            )
            writer.write_table(table)
            kept += table.num_rows
            if verbose and position % 5 == 0:
                print("transactions chunk %d rows kept %d users %d" % (position, kept, len(index)))
    finally:
        writer.close()

    index.frame().to_parquet(users_target, index=False, compression="zstd")
    summary = {
        "rows_read": seen,
        "rows_kept": kept,
        "users": len(index),
        "dropped_bad_dates": dropped_dates,
        "dropped_outside_window": dropped_window,
    }
    if verbose:
        print(summary)
    return summary


def ingest_members(config, verbose=True):
    source = config.paths.raw_dir / config.data.members_file
    if not source.is_file():
        raise FileNotFoundError("no members file at %s" % source)
    users = pd.read_parquet(config.paths.interim_dir / "users.parquet")
    lookup = pd.Index(users["msno"].to_numpy())
    target = config.paths.interim_dir / "members.parquet"

    frames = []
    matched = 0
    seen = 0
    reader = pd.read_csv(
        source,
        dtype=MEMBER_DTYPES,
        chunksize=config.data.chunk_rows,
        usecols=list(MEMBER_DTYPES),
        keep_default_na=False,
    )
    for chunk in reader:
        seen += len(chunk)
        codes = lookup.get_indexer(chunk["msno"].to_numpy(dtype=object))
        keep = codes >= 0
        if not keep.any():
            continue
        chunk = chunk.loc[keep]
        registration_day, ok = yyyymmdd_to_day(chunk["registration_init_time"].to_numpy())
        frames.append(
            pd.DataFrame(
                {
                    "user_id": codes[keep].astype(np.int32),
                    "city": chunk["city"].to_numpy().astype(np.int16),
                    "bd": chunk["bd"].to_numpy().astype(np.int32),
                    "gender_code": chunk["gender"]
                    .fillna("")
                    .map(GENDER_CODES)
                    .fillna(0)
                    .to_numpy()
                    .astype(np.int8),
                    "registered_via": chunk["registered_via"].to_numpy().astype(np.int16),
                    "registration_day": np.where(ok, registration_day, -1).astype(np.int32),
                }
            )
        )
        matched += int(keep.sum())

    if frames:
        members = pd.concat(frames, ignore_index=True).drop_duplicates("user_id", keep="last")
    else:
        members = pd.DataFrame(
            {
                "user_id": np.array([], dtype=np.int32),
                "city": np.array([], dtype=np.int16),
                "bd": np.array([], dtype=np.int32),
                "gender_code": np.array([], dtype=np.int8),
                "registered_via": np.array([], dtype=np.int16),
                "registration_day": np.array([], dtype=np.int32),
            }
        )
    members.to_parquet(target, index=False, compression="zstd")
    summary = {"rows_read": seen, "rows_matched": len(members), "users_total": len(users)}
    if verbose:
        print(summary)
    return summary


def run(config, verbose=True):
    config.paths.ensure()
    transactions = ingest_transactions(config, verbose=verbose)
    members = ingest_members(config, verbose=verbose)
    return {"transactions": transactions, "members": members}
