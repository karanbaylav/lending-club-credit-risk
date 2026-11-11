"""Default definition, exit ages and the 12-month target on a hand-made raw table."""
import duckdb
import pytest

from src.run import run_file

COLS = ["id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
        "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status",
        "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
        "pub_rec_bankruptcies", "application_type", "initial_list_status", "policy_code", "total_pymnt",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "out_prncp", "last_pymnt_d"]

# id, status, issue, last payment
ROWS = [
    ("1", "Fully Paid", "Jan-2015", "Jan-2018"),
    ("2", "Charged Off", "Jan-2015", "Jun-2015"),                      # default in month 6
    ("3", "Charged Off", "Jan-2015", None),                            # never paid: default month 1
    ("4", "Current", "Jan-2015", "Mar-2019"),                          # open, old: 12m outcome known (non-default)
    ("5", "Current", "Jan-2018", "Mar-2019"),                          # open, young: 12m outcome unknown
    ("6", "Does not meet the credit policy. Status:Charged Off", "Jan-2010", "Jun-2011"),  # default month 18
    ("7", "Late (31-120 days)", "Jan-2016", "Dec-2018"),
]


@pytest.fixture()
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE raw_accepted (" + ", ".join(f'"{x}" VARCHAR' for x in COLS) + ")")
    for i, st, iss, lp in ROWS:
        vals = {c_: None for c_ in COLS}
        vals.update(id=i, loan_amnt="10000", funded_amnt="10000", term=" 36 months", int_rate="12.5",
                    installment="334", grade="B", sub_grade="B3", issue_d=iss, loan_status=st, last_pymnt_d=lp,
                    fico_range_low="690", fico_range_high="694", earliest_cr_line="Jan-2005", annual_inc="50000",
                    total_pymnt="5000")
        c.execute("INSERT INTO raw_accepted VALUES (" + ",".join("?" * len(COLS)) + ")", [vals[k] for k in COLS])
    c.execute("INSERT INTO raw_accepted (id) VALUES ('Total amount funded in policy code 1: 123')")  # footer row
    run_file(c, "sql/02_clean.sql", save=False)
    return c


def get(con, loan_id, col):
    return con.execute(f"SELECT {col} FROM loan_events WHERE loan_id = {loan_id}").fetchone()[0]


def test_footer_dropped(con):
    assert con.execute("SELECT COUNT(*) FROM loans_typed").fetchone()[0] == len(ROWS)


def test_status_groups(con):
    assert get(con, 1, "status_group") == "paid"
    assert get(con, 2, "status_group") == "default"
    assert get(con, 6, "status_group") == "default" and get(con, 6, "policy_fail")
    assert get(con, 7, "status_group") == "open"


def test_default_month(con):
    assert get(con, 2, "exit_age") == 6
    assert get(con, 3, "exit_age") == 1
    assert get(con, 6, "exit_age") == 18


def test_12m_target(con):
    assert get(con, 2, "bad_12m") and get(con, 3, "bad_12m")
    assert not get(con, 6, "bad_12m")          # defaulted, but after month 12
    assert get(con, 4, "obs_12m") and not get(con, 4, "bad_12m")
    assert not get(con, 5, "obs_12m")          # too young at the Mar-2019 snapshot


def test_open_loans_not_resolved(con):
    ids = {r[0] for r in con.execute("SELECT loan_id FROM loans_resolved").fetchall()}
    assert ids == {1, 2, 3, 6}
