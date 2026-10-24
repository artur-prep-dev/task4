import json, re, sqlite3
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yaml

FOLDER = Path(__file__).parent / "data"
AI_URL = "https://api.openai.com/v1/chat/completions"
FIELDS = ["name", "address", "phone", "email"]

def get_price(text):
    text = str(text)
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    if len(nums) > 1:
        price = float(nums[0] + "." + nums[1])
    else:
        price = float(nums[0])
    if "€" in text or "EUR" in text.upper():
        price = price * 1.2
    return price

def get_dates(col):
    col = col.str.upper().str.replace(".M.", "M")
    col = col.str.replace(";", " ").str.replace(",", " ")
    col = col.str.replace(r"(\d+)\.(\d+)\.(\d{4})", r"\3-\2-\1", regex=True)
    return pd.to_datetime(col, format="mixed", errors="coerce")

def sort_authors(text):
    names = [n.strip() for n in str(text).split(",")]
    return ", ".join(sorted(names))

def fix_year(year):
    if pd.isna(year) or year <= 0:
        return None
    return int(year)

def clean(value):
    if pd.isna(value):
        return None
    return str(value).lower()

def find_people(users):
    rows = []
    for _, user in users.iterrows():
        phone = None
        if not pd.isna(user["phone"]):
            phone = re.sub(r"\D", "", user["phone"]) or None
        rows.append([clean(user["name"]), clean(user["address"]), phone, clean(user["email"])])
    person = users["id"].tolist()
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            same = 0
            for k in range(4):
                if rows[i][k] is not None and rows[i][k] == rows[j][k]:
                    same += 1
            if same >= 3 and person[i] != person[j]:
                new = min(person[i], person[j])
                old = max(person[i], person[j])
                for k in range(len(person)):
                    if person[k] == old:
                        person[k] = new
    return person

def load_store(folder):
    users = pd.read_csv(folder / "users.csv", dtype=str)
    for f in FIELDS:
        users[f] = users[f].str.strip().replace("", pd.NA)
    users = users.drop_duplicates().dropna(subset=["id"]).reset_index(drop=True)
    users["id"] = users["id"].astype(int)
    users["person"] = find_people(users)
    with open(folder / "books.yaml", encoding="utf-8") as file:
        books = pd.DataFrame(yaml.safe_load(file))
    books.columns = books.columns.str.lstrip(":")
    books = books.drop_duplicates(subset="id")
    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    books["year"] = books["year"].apply(fix_year).astype("Int64")
    books["authors"] = books["author"].apply(sort_authors)
    orders = pd.read_parquet(folder / "orders.parquet").drop_duplicates()
    orders["shipping"] = orders["shipping"].replace(["", "NULL"], pd.NA)
    orders["unit_price"] = orders["unit_price"].apply(get_price)
    orders["timestamp"] = get_dates(orders["timestamp"])
    orders = orders.dropna(subset=["unit_price", "timestamp"])
    orders = orders[orders["quantity"] > 0]
    orders = orders[orders["user_id"].isin(users["id"]) & orders["book_id"].isin(books["id"])]
    orders["paid_price"] = (orders["quantity"] * orders["unit_price"]).round(2)
    orders["date"] = orders["timestamp"].dt.date
    people = users[["id", "person"]].rename(columns={"id": "user_id"})
    authors = books[["id", "authors"]].rename(columns={"id": "book_id"})
    orders = orders.merge(people, on="user_id").merge(authors, on="book_id")
    return {"users": users, "books": books, "orders": orders}

@st.cache_data
def load_all():
    stores = {}
    for folder in sorted(FOLDER.glob("STORE*")):
        stores[folder.name] = load_store(folder)
    return stores

def get_answers(users, books, orders, all_users):
    days = orders.groupby("date")["paid_price"].sum().round(2).reset_index()
    days.columns = ["date", "money"]
    top5 = days.sort_values("money", ascending=False).head(5).copy()
    top5["date"] = top5["date"].astype(str)

    sold = orders.groupby("authors")["quantity"].sum()
    top_authors = sold[sold == sold.max()].index.tolist()
    spent = orders.groupby("person")["paid_price"].sum()
    best_ids = all_users[all_users["person"] == spent.idxmax()]["id"].tolist()
    return {
        "days": days,
        "top5": top5.to_dict("records"),
        "people": int(users["person"].nunique()),
        "author_sets": int(books["authors"].nunique()),
        "top_authors": top_authors,
        "top_sold": int(sold.max()),
        "best_buyer_ids": sorted(best_ids),
        "best_buyer_spent": round(float(spent.max()), 2),
        "books_sold": int(orders["quantity"].sum()),
        "money": round(float(orders["paid_price"].sum()), 2),
    }

def empty_filters(store):
    return {"store": store, "start": None, "end": None, "chart": "line",
            "hidden_books": [], "emails": [], "marked": []}

def email_starts_with(users, emails):
    result = []
    for email in users["email"]:
        found = False
        if not pd.isna(email):
            for start in emails:
                if email.lower().startswith(start.lower()):
                    found = True
        result.append(found)
    return pd.Series(result, index=users.index)

def apply_filters(data, f):
    users, books, orders = data["users"], data["books"], data["orders"]
    if f["hidden_books"]:
        books = books[~books["id"].isin(f["hidden_books"])]
        orders = orders[orders["book_id"].isin(books["id"])]
    if f["emails"]:
        users = users[email_starts_with(users, f["emails"])]
        orders = orders[orders["user_id"].isin(users["id"])]
    if f["start"]:
        orders = orders[orders["date"] >= pd.to_datetime(f["start"]).date()]
    if f["end"]:
        orders = orders[orders["date"] <= pd.to_datetime(f["end"]).date()]
    return users, books, orders

PROMPT = """You are a helper in a dashboard of book stores.
Answer in the user's language. Use only the numbers given here or SQL results, never guess.
Reply ONLY with JSON like this:
{"answer": "text for the user", "sql": "", "filters": {}}
"sql": if the numbers below are not enough, write one SQLite SELECT query and leave "answer" empty.
SQL runs on the current store. You will get the result and then write the answer.
"filters": to change the dashboard, put only the keys you want to change:
"store": "STORE1".."STORE5", "start" and "end": "YYYY-MM-DD" or "", "chart": "line" or "bar",
"hidden_books": [book ids], "emails": [email beginnings to show], "marked": [email beginnings to mark].
To remove all filters: {"clear": true}.
"""

def run_sql(data, sql):
    if sql.split()[0].lower() not in ["select", "with"]:
        return {"error": "Only SELECT queries are allowed."}
    orders = data["orders"].copy()
    orders["date"] = orders["date"].astype(str)
    db = sqlite3.connect(":memory:")
    data["users"].to_sql("users", db, index=False)
    data["books"].to_sql("books", db, index=False)
    orders.to_sql("orders", db, index=False)
    try:
        res = pd.read_sql(sql, db)
    except Exception as e:
        return {"error": str(e)}
    finally:
        db.close()
    return {"rows": res.head(100).to_dict("records"), "total_rows": len(res)}

def change_filters(new, stores, f):
    if not new:
        return
    if new.get("clear"):
        f.update(empty_filters(f["store"]))
    if new.get("store") in stores and new["store"] != f["store"]:
        f.update(empty_filters(new["store"]))
    for name in ["start", "end", "chart", "hidden_books", "emails", "marked"]:
        if name in new:
            f[name] = new[name]
    f["start"] = f["start"] or None
    f["end"] = f["end"] or None
    if f["chart"] not in ["line", "bar"]:
        f["chart"] = "line"
    f["hidden_books"] = [int(b) for b in f["hidden_books"]]
    f["emails"] = [e.lower() for e in f["emails"]]
    f["marked"] = [e.lower() for e in f["marked"]]

def send(msgs, key, model):
    r = requests.post(AI_URL, json={"model": model, "messages": msgs},
                      headers={"Authorization": "Bearer " + key}, timeout=120)
    if r.status_code != 200:
        raise Exception(f"AI error {r.status_code}: {r.text[:400]}")
    text = r.json()["choices"][0]["message"]["content"] or ""
    msgs.append({"role": "assistant", "content": text})
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"answer": text}

def ask_ai(question, stores, f, key, model):
    numbers = {}
    for name in stores:
        data = stores[name]
        res = get_answers(data["users"], data["books"], data["orders"], data["users"])
        del res["days"]
        numbers[name] = res
    data = stores[f["store"]]
    columns = {}
    for table in ["users", "books", "orders"]:
        columns[table] = list(data[table].columns)
    info = ("Main numbers: " + json.dumps(numbers) + "\nTable columns: " + json.dumps(columns)
            + "\nCurrent filters: " + json.dumps(f))
    msgs = [{"role": "system", "content": PROMPT + info}]
    for m in st.session_state["chat"][-10:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": question})
    used = []
    reply = send(msgs, key, model)
    change_filters(reply.get("filters"), stores, f)
    if reply.get("sql"):
        result = run_sql(stores[f["store"]], reply["sql"])
        used.append({"sql": reply["sql"], "result": result})
        msgs.append({"role": "user", "content": "SQL result: " + json.dumps(result, default=str)})
        reply = send(msgs, key, model)
        change_filters(reply.get("filters"), stores, f)
    used.append({"filters now": f})
    return reply.get("answer") or "Done.", used

def reload():
    st.session_state["version"] += 1
    st.rerun()

def sidebar(stores, f):
    v = st.session_state["version"]
    names = list(stores)
    store = st.sidebar.selectbox("Store", names, names.index(f["store"]), key=f"s{v}")
    if store != f["store"]:
        f.update(empty_filters(store))
        reload()
    f["chart"] = st.sidebar.radio("Chart", ["line", "bar"], ["line", "bar"].index(f["chart"]), key=f"c{v}")
    emails = st.sidebar.text_input("Email starts with (a, b, ...)", ", ".join(f["emails"]), key=f"e{v}")
    f["emails"] = [e.strip().lower() for e in emails.split(",") if e.strip()]
    if st.sidebar.button("Clear filters"):
        f.update(empty_filters(store))
        reload()
    model = st.sidebar.text_input("AI model", "gpt-6-luna")
    key = st.sidebar.text_input("OpenAI API key", type="password")
    if not key:
        try:
            key = st.secrets["LLM_API_KEY"]
        except Exception:
            pass
    return store, model, key

def chat(stores, f, key, model):
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("used"):
                with st.expander("Details"):
                    st.json(m["used"])
    if not key:
        st.info("Paste your OpenAI API key on the left.")
        return
    question = st.chat_input("Ask about the data or give a command")
    if question:
        with st.spinner("Thinking..."):
            try:
                answer, used = ask_ai(question, stores, f, key, model)
            except Exception as e:
                st.error(str(e))
                return
        st.session_state["chat"].append({"role": "user", "content": question})
        st.session_state["chat"].append({"role": "assistant", "content": answer, "used": used})
        reload()

def main():
    st.set_page_config(page_title="Book Stores", layout="wide")
    stores = load_all()
    if "filters" not in st.session_state:
        st.session_state["filters"] = empty_filters(list(stores)[0])
        st.session_state["chat"] = []
        st.session_state["version"] = 0
    f = st.session_state["filters"]
    store, model, key = sidebar(stores, f)
    data = stores[store]
    users, books, orders = apply_filters(data, f)

    st.title(f"{store} - Book sales dashboard")
    st.caption(f"Dates: {f['start'] or 'all'} to {f['end'] or 'all'} | Hidden books: {len(f['hidden_books'])} | "
               f"Emails: {', '.join(f['emails']) or 'all'} | Prices in USD (1 EUR = 1.2 USD)")
    page = st.radio("Page", ["Revenue", "Customers", "Authors", "AI chat"], horizontal=True, key="page")
    if page == "AI chat":
        chat(stores, f, key, model)
        return
    if orders.empty:
        st.warning("No orders for these filters. Press 'Clear filters'.")
        return
    a = get_answers(users, books, orders, data["users"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${a['money']:,.2f}")
    c2.metric("Real unique users", f"{a['people']:,}")
    c3.metric("Unique author sets", f"{a['author_sets']:,}")
    c4.metric("Books sold", f"{a['books_sold']:,}")
    if page == "Revenue":
        if f["chart"] == "bar":
            chart = px.bar(a["days"], x="date", y="money", title="Daily revenue, USD")
        else:
            chart = px.line(a["days"], x="date", y="money", title="Daily revenue, USD")
        st.plotly_chart(chart)
        st.subheader("Top 5 days by revenue")
        top = pd.DataFrame(a["top5"])
        top_chart = px.bar(top, x="date", y="money", text="money")
        top_chart.update_xaxes(type="category")
        st.plotly_chart(top_chart)
        st.dataframe(top, hide_index=True)
    if page == "Customers":
        st.subheader("Best buyer (all ids)")
        st.code(str(a["best_buyer_ids"]))
        st.caption(f"Spent: ${a['best_buyer_spent']:,.2f}")
        table = users.copy()
        table.insert(0, "marked", email_starts_with(table, f["marked"]))
        st.subheader(f"Users ({table['marked'].sum()} marked)")
        st.dataframe(table.sort_values("marked", ascending=False), hide_index=True)
    if page == "Authors":
        st.subheader("Most popular author(s)")
        for name in a["top_authors"]:
            st.markdown(f"### {name}")
        st.caption(f"{a['top_sold']:,} books sold")
        hidden = data["books"][data["books"]["id"].isin(f["hidden_books"])]
        if len(hidden) > 0:
            st.subheader("Hidden books")
            st.dataframe(hidden[["id", "title", "authors"]], hide_index=True)

main()
