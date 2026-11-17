from __future__ import annotations

import base64

import numpy as np
import pandas as pd

PLAN_DAYS = np.array([7, 30, 90, 100, 180, 195, 410, 450], dtype=np.int32)
PLAN_WEIGHTS = np.array([0.020, 0.855, 0.040, 0.010, 0.030, 0.010, 0.023, 0.012])
PLAN_PRICE = np.array([49, 149, 417, 465, 788, 894, 1788, 1900], dtype=np.int32)

METHOD_IDS = np.array([41, 40, 36, 38, 39, 37, 34, 29, 30, 32, 22, 12], dtype=np.int32)
METHOD_WEIGHTS = np.array(
    [0.520, 0.115, 0.085, 0.070, 0.055, 0.040, 0.030, 0.025, 0.020, 0.018, 0.014, 0.008]
)
METHOD_RISK = np.array(
    [-0.35, -0.10, 0.05, 0.20, 0.15, 0.30, 0.45, 0.55, 0.40, 0.25, 0.70, 0.85]
)
METHOD_AUTO_RENEW_BIAS = np.array(
    [0.30, 0.15, 0.00, -0.20, -0.10, -0.30, -0.45, -0.55, -0.40, -0.25, -0.70, -0.90]
)

CITY_IDS = np.array([1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 22], dtype=np.int32)
CITY_WEIGHTS = np.array(
    [
        0.470, 0.070, 0.075, 0.055, 0.045, 0.040, 0.035, 0.030,
        0.028, 0.025, 0.030, 0.022, 0.020, 0.020, 0.020, 0.015,
    ]
)
REGISTERED_VIA = np.array([3, 4, 7, 9, 13], dtype=np.int32)
REGISTERED_VIA_WEIGHTS = np.array([0.28, 0.22, 0.36, 0.09, 0.05])

EPOCH = np.datetime64("1970-01-01")


def _to_yyyymmdd(days):
    stamps = EPOCH + days.astype("timedelta64[D]")
    years = stamps.astype("datetime64[Y]").astype(np.int64) + 1970
    months = stamps.astype("datetime64[M]").astype(np.int64) % 12 + 1
    day_of_month = (stamps - stamps.astype("datetime64[M]")).astype(np.int64) + 1
    return (years * 10000 + months * 100 + day_of_month).astype(np.int64)


def _day_index(value):
    return int((np.datetime64(str(value)) - EPOCH).astype(int))


def _make_ids(rng, count):
    payload = rng.integers(0, 256, size=count * 33, dtype=np.uint8).tobytes()
    encoded = base64.b64encode(payload).decode("ascii")
    ids = np.array([encoded[i * 44 : i * 44 + 43] + "=" for i in range(count)], dtype=object)
    return ids


def _choice(rng, values, weights, size):
    picks = rng.choice(len(values), size=size, p=weights / weights.sum())
    return picks


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _member_frame(rng, ids, settings):
    count = len(ids)
    city_idx = _choice(rng, CITY_IDS, CITY_WEIGHTS, count)
    via_idx = _choice(rng, REGISTERED_VIA, REGISTERED_VIA_WEIGHTS, count)
    age = rng.normal(27.5, 9.0, size=count)
    junk = rng.random(count)
    age = np.where(junk < 0.42, 0.0, age)
    age = np.where((junk >= 0.42) & (junk < 0.445), rng.integers(-40, 0, size=count), age)
    age = np.where((junk >= 0.445) & (junk < 0.455), rng.integers(120, 2020, size=count), age)
    gender_draw = rng.random(count)
    gender = np.where(gender_draw < 0.40, "", np.where(gender_draw < 0.70, "male", "female"))
    return pd.DataFrame(
        {
            "msno": ids,
            "city": CITY_IDS[city_idx],
            "bd": np.rint(age).astype(np.int64),
            "gender": gender,
            "registered_via": REGISTERED_VIA[via_idx],
            "registration_init_time": None,
        }
    )


def generate(config, transactions_path=None, members_path=None, verbose=True):
    settings = config.synthetic
    rng = np.random.default_rng(settings.seed)
    n_users = settings.n_users
    raw_dir = config.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    transactions_path = transactions_path or raw_dir / config.data.transactions_file
    members_path = members_path or raw_dir / config.data.members_file

    signup_start = _day_index(settings.signup_start)
    signup_end = _day_index(settings.signup_end)
    horizon_end = _day_index(config.data.observation_end)

    ids = _make_ids(rng, n_users)
    members = _member_frame(rng, ids, settings)

    signup = rng.integers(signup_start, signup_end + 1, size=n_users).astype(np.int64)
    registration = signup - rng.integers(0, 900, size=n_users).astype(np.int64)
    registration = np.maximum(registration, signup_start - 1800)
    members["registration_init_time"] = _to_yyyymmdd(registration)

    plan_idx = _choice(rng, PLAN_DAYS, PLAN_WEIGHTS, n_users)
    method_idx = _choice(rng, METHOD_IDS, METHOD_WEIGHTS, n_users)
    auto_logit = (
        np.log(settings.auto_renew_rate / (1.0 - settings.auto_renew_rate))
        + METHOD_AUTO_RENEW_BIAS[method_idx]
    )
    auto_renew = (rng.random(n_users) < _sigmoid(auto_logit)).astype(np.int8)

    frailty = rng.normal(0.0, 0.55, size=n_users)
    city_pos = np.searchsorted(CITY_IDS, members["city"].to_numpy())
    city_effect = np.where(city_pos == 0, -0.12, 0.08)

    plan_days = PLAN_DAYS[plan_idx].astype(np.int64)
    list_price = PLAN_PRICE[plan_idx].astype(np.int64)
    paid = list_price.copy()
    method = METHOD_IDS[method_idx].astype(np.int64)
    method_pos = method_idx.copy()

    txn_date = signup.copy()
    expire = txn_date + plan_days
    active = np.ones(n_users, dtype=bool)
    cycle = np.zeros(n_users, dtype=np.int64)
    had_discount = np.zeros(n_users, dtype=bool)
    price_rose = np.zeros(n_users, dtype=bool)

    base_logit = np.log(settings.base_monthly_hazard / (1.0 - settings.base_monthly_hazard))
    log_auto = np.log(settings.auto_renew_hazard_multiplier)
    log_price = np.log(settings.price_increase_hazard_multiplier)
    log_discount = np.log(settings.discount_hazard_multiplier)

    if transactions_path.exists():
        transactions_path.unlink()

    header = True
    rows_written = 0
    first = pd.DataFrame(
        {
            "msno": ids,
            "payment_method_id": method,
            "payment_plan_days": plan_days,
            "plan_list_price": list_price,
            "actual_amount_paid": paid,
            "is_auto_renew": auto_renew.astype(np.int64),
            "transaction_date": _to_yyyymmdd(txn_date),
            "membership_expire_date": _to_yyyymmdd(expire),
            "is_cancel": np.zeros(n_users, dtype=np.int64),
        }
    )
    first.sample(frac=1.0, random_state=int(settings.seed % 2**31)).to_csv(
        transactions_path, index=False, mode="w", header=header
    )
    rows_written += len(first)
    header = False

    max_cycles = 400
    for _ in range(max_cycles):
        due = active & (expire <= horizon_end)
        if not due.any():
            break
        idx = np.flatnonzero(due)

        hazard_logit = (
            base_logit
            + log_auto * auto_renew[idx]
            + log_price * price_rose[idx]
            + log_discount * had_discount[idx]
            - settings.tenure_hazard_decay * cycle[idx]
            + METHOD_RISK[method_pos[idx]]
            + frailty[idx]
            + city_effect[idx]
        )
        monthly = _sigmoid(hazard_logit)
        scaled = 1.0 - np.power(1.0 - monthly, plan_days[idx] / 30.0)
        churn = rng.random(idx.size) < scaled

        share = np.where(
            auto_renew[idx] == 1,
            settings.voluntary_share_auto_renew,
            settings.voluntary_share_manual,
        )
        voluntary = churn & (rng.random(idx.size) < share)

        renew_idx = idx[~churn]
        cancel_idx = idx[voluntary]
        lapse_idx = idx[churn & ~voluntary]

        frames = []

        if cancel_idx.size:
            offset = rng.integers(1, np.maximum(plan_days[cancel_idx] - 1, 2))
            cancel_date = np.minimum(expire[cancel_idx] - offset, horizon_end)
            cancel_date = np.maximum(cancel_date, txn_date[cancel_idx] + 1)
            keeps_term = rng.random(cancel_idx.size) < 0.6
            new_expire = np.where(keeps_term, expire[cancel_idx], cancel_date)
            frames.append(
                pd.DataFrame(
                    {
                        "msno": ids[cancel_idx],
                        "payment_method_id": method[cancel_idx],
                        "payment_plan_days": plan_days[cancel_idx],
                        "plan_list_price": list_price[cancel_idx],
                        "actual_amount_paid": np.zeros(cancel_idx.size, dtype=np.int64),
                        "is_auto_renew": auto_renew[cancel_idx].astype(np.int64),
                        "transaction_date": _to_yyyymmdd(cancel_date),
                        "membership_expire_date": _to_yyyymmdd(new_expire),
                        "is_cancel": np.ones(cancel_idx.size, dtype=np.int64),
                    }
                )
            )
            expire[cancel_idx] = new_expire

        if renew_idx.size:
            jitter = rng.integers(-3, 3, size=renew_idx.size)
            new_txn = np.minimum(expire[renew_idx] + jitter, horizon_end)
            new_txn = np.maximum(new_txn, txn_date[renew_idx] + 1)

            switch_plan = rng.random(renew_idx.size) < 0.025
            if switch_plan.any():
                targets = renew_idx[switch_plan]
                new_plan_idx = _choice(rng, PLAN_DAYS, PLAN_WEIGHTS, targets.size)
                plan_days[targets] = PLAN_DAYS[new_plan_idx]
                list_price[targets] = PLAN_PRICE[new_plan_idx]

            switch_method = rng.random(renew_idx.size) < 0.035
            if switch_method.any():
                targets = renew_idx[switch_method]
                new_method_idx = _choice(rng, METHOD_IDS, METHOD_WEIGHTS, targets.size)
                method[targets] = METHOD_IDS[new_method_idx]
                method_pos[targets] = new_method_idx

            toggle = rng.random(renew_idx.size) < 0.02
            if toggle.any():
                targets = renew_idx[toggle]
                auto_renew[targets] = 1 - auto_renew[targets]

            bump = rng.random(renew_idx.size) < 0.03
            previous_price = list_price[renew_idx].copy()
            if bump.any():
                targets = renew_idx[bump]
                direction = np.where(rng.random(targets.size) < 0.7, 1.0, -1.0)
                list_price[targets] = np.maximum(
                    np.rint(list_price[targets] * (1.0 + direction * 0.10)).astype(np.int64), 1
                )
            price_rose[renew_idx] = list_price[renew_idx] > previous_price

            discount = rng.random(renew_idx.size) < 0.055
            new_paid = list_price[renew_idx].copy()
            if discount.any():
                targets = np.flatnonzero(discount)
                new_paid[targets] = np.maximum(
                    np.rint(new_paid[targets] * rng.uniform(0.0, 0.8, size=targets.size)).astype(
                        np.int64
                    ),
                    0,
                )
            paid[renew_idx] = new_paid
            had_discount[renew_idx] = new_paid < list_price[renew_idx]

            new_expire = np.maximum(expire[renew_idx], new_txn) + plan_days[renew_idx]
            frames.append(
                pd.DataFrame(
                    {
                        "msno": ids[renew_idx],
                        "payment_method_id": method[renew_idx],
                        "payment_plan_days": plan_days[renew_idx],
                        "plan_list_price": list_price[renew_idx],
                        "actual_amount_paid": paid[renew_idx],
                        "is_auto_renew": auto_renew[renew_idx].astype(np.int64),
                        "transaction_date": _to_yyyymmdd(new_txn),
                        "membership_expire_date": _to_yyyymmdd(new_expire),
                        "is_cancel": np.zeros(renew_idx.size, dtype=np.int64),
                    }
                )
            )
            txn_date[renew_idx] = new_txn
            expire[renew_idx] = new_expire
            cycle[renew_idx] += 1

        active[cancel_idx] = False
        active[lapse_idx] = False

        if frames:
            batch = pd.concat(frames, ignore_index=True)
            batch = batch.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
            batch.to_csv(transactions_path, index=False, mode="a", header=header)
            rows_written += len(batch)

    keep = rng.random(n_users) >= settings.members_missing_rate
    members.loc[keep].sample(
        frac=1.0, random_state=int(settings.seed % 2**31)
    ).to_csv(members_path, index=False)

    if verbose:
        print("users %d" % n_users)
        print("transactions %d" % rows_written)
        print("members %d" % int(keep.sum()))
        print("still active at end %d" % int(active.sum()))
    return {
        "users": n_users,
        "transactions": rows_written,
        "members": int(keep.sum()),
        "active_at_end": int(active.sum()),
    }
