from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42
random.seed(SEED)

OUT_DIR = Path("Guvenlik/web_attack/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "web_attack_v2_sql_hardneg.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def benign_sql_examples() -> list[str]:
    examples: list[str] = []

    # Safe SQL fragments. These are not attacks by themselves.
    examples.extend(
        [
            "WHERE",
            "SELECT",
            "AND",
            "OR",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "JOIN",
            "LEFT JOIN",
            "INNER JOIN",
            "IS NOT NULL",
            "IS NULL",
            "COUNT(*)",
            "COUNT(DISTINCT order_id)",
            "DATE '2024-01-01'",
            "CURRENT_DATE - INTERVAL '30 days'",
            "status IN ('completed', 'shipped', 'delivered')",
            "created_at >= DATE '2023-01-01'",
            "created_at < DATE '2026-01-01'",
            "email IS NOT NULL",
            "COUNT(DISTINCT o.order_id) >= 0",
            "ELSE 'STANDARD'",
            "WHEN total_amount >= 10000 THEN 'VIP'",
        ]
    )

    # Safe SELECT queries.
    examples.extend(
        [
            "SELECT * FROM users;",
            "SELECT id FROM users;",
            "SELECT id, name FROM users;",
            "SELECT id, name FROM users WHERE id = 5;",
            "SELECT email FROM users WHERE is_active = TRUE;",
            "SELECT COUNT(*) FROM orders;",
            "SELECT COUNT(*) FROM orders GROUP BY customer_id;",
            "SELECT AVG(total_amount) FROM orders;",
            "SELECT product_name FROM products ORDER BY created_at DESC;",
            "SELECT gender_code, COUNT(*) FROM Guests GROUP BY gender_code ORDER BY COUNT(*) DESC;",
            "SELECT claim_status_description FROM claims_processing_stages WHERE claim_status_name = \"Open\";",
            "SELECT AVG(Snatch) FROM body_builder;",
            "SELECT name FROM products WHERE category_id = 3;",
            "SELECT id, email FROM customers WHERE email IS NOT NULL;",
            "SELECT username FROM users WHERE role = 'admin';",
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';",
        ]
    )

    # Safe write/delete SQL statements. They can be policy-sensitive, but they are not SQL injection by themselves.
    examples.extend(
        [
            "INSERT INTO logs(event_type, created_at) VALUES ('login_success', NOW());",
            "INSERT INTO audit_logs(user_id, action, created_at) VALUES (10, 'profile_update', CURRENT_TIMESTAMP);",
            "INSERT INTO users(name, email, created_at) VALUES ('Mehmet', 'mehmet@example.com', CURRENT_TIMESTAMP);",
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = 10;",
            "UPDATE orders SET status = 'completed' WHERE id = 42;",
            "UPDATE products SET stock_count = stock_count - 1 WHERE product_id = 15;",
            "DELETE FROM sessions WHERE expires_at < NOW();",
            "DELETE FROM temporary_tokens WHERE created_at < CURRENT_DATE - INTERVAL '7 days';",
            "DELETE FROM cart_items WHERE cart_id = 100 AND product_id = 25;",
        ]
    )

    tables = [
        "users",
        "customers",
        "orders",
        "products",
        "payments",
        "invoices",
        "employees",
        "sessions",
        "audit_logs",
        "security_logs",
    ]

    statuses = [
        "('completed', 'shipped', 'delivered')",
        "('active', 'pending', 'verified')",
        "('paid', 'refunded', 'cancelled')",
        "('login_success', 'login_failed', 'password_reset')",
    ]

    date_cols = ["created_at", "updated_at", "order_date", "payment_date", "expires_at"]

    # Medium-length safe SELECT queries.
    for _ in range(1500):
        table = random.choice(tables)
        date_col = random.choice(date_cols)
        status_list = random.choice(statuses)

        examples.append(
            f"""
SELECT
    id,
    name,
    created_at,
    updated_at
FROM {table}
WHERE
    is_active = TRUE
    AND {date_col} >= DATE '2023-01-01'
    AND {date_col} < DATE '2026-01-01'
    AND status IN {status_list}
ORDER BY
    created_at DESC
LIMIT 100;
""".strip()
        )

    # Safe JOIN / GROUP BY / HAVING queries.
    for _ in range(1500):
        examples.append(
            """
SELECT
    c.customer_id,
    c.full_name,
    c.email,
    COUNT(DISTINCT o.order_id) AS total_order_count,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_order_amount,
    MAX(o.created_at) AS last_order_date,
    CASE
        WHEN COUNT(DISTINCT o.order_id) = 0 THEN 'NO_ORDER'
        WHEN COALESCE(SUM(oi.quantity * oi.unit_price), 0) >= 10000 THEN 'VIP'
        WHEN COALESCE(SUM(oi.quantity * oi.unit_price), 0) >= 3000 THEN 'LOYAL'
        ELSE 'STANDARD'
    END AS customer_segment
FROM customers c
LEFT JOIN orders o
    ON o.customer_id = c.customer_id
    AND o.status IN ('completed', 'shipped', 'delivered')
LEFT JOIN order_items oi
    ON oi.order_id = o.order_id
WHERE
    c.is_active = TRUE
    AND c.created_at >= DATE '2023-01-01'
    AND c.created_at < DATE '2026-01-01'
    AND c.email IS NOT NULL
GROUP BY
    c.customer_id,
    c.full_name,
    c.email
HAVING
    COUNT(DISTINCT o.order_id) >= 0
ORDER BY
    total_order_amount DESC,
    last_order_date DESC
LIMIT 250;
""".strip()
        )

    # Security-related but safe reporting queries.
    for _ in range(700):
        examples.append(
            """
SELECT
    event_type,
    COUNT(*) AS event_count
FROM security_logs
WHERE
    event_type IN ('login_success', 'login_failed', 'password_reset')
    AND created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY
    event_type
ORDER BY
    event_count DESC;
""".strip()
        )

    # Safe INSERT/UPDATE/DELETE variants.
    for _ in range(600):
        uid = random.randint(1, 10000)
        examples.append(f"UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = {uid};")
        examples.append(f"DELETE FROM sessions WHERE user_id = {uid} AND expires_at < NOW();")
        examples.append(f"INSERT INTO logs(user_id, event_type, created_at) VALUES ({uid}, 'login_success', NOW());")

    return examples


def hard_attack_examples() -> list[str]:
    return [
        # Classic SQLi
        "login=admin' OR '1'='1' --&password=test&form=submit",
        "' OR '1'='1' --",
        "' OR 1=1--",
        "admin' --",
        "admin' #",
        "id=1' AND 1=1--",
        "id=1' AND 1=2--",

        # UNION-based SQLi
        "movie=1 UNION SELECT NULL,username,password,email,secret,NULL,NULL FROM users--",
        "id=1 UNION SELECT username,password FROM users--",
        "id=1' UNION SELECT table_name,column_name FROM information_schema.columns--",

        # Error-based SQLi
        "id=1' AND EXTRACTVALUE(1, CONCAT(0x7e, DATABASE(), 0x7e))--",
        "id=1' AND UPDATEXML(1, CONCAT(0x7e, USER(), 0x7e), 1)--",

        # Time-based / boolean blind SQLi
        "id=1' AND SLEEP(5)--",
        "id=1' AND IF(SUBSTRING(DATABASE(),1,1)='a',SLEEP(5),0)--",
        "id=1' AND BENCHMARK(5000000,MD5(1))--",

        # Stacked query / destructive injection
        "id=1; DROP TABLE users;--",
        "id=1; DELETE FROM users WHERE role='admin';--",

        # Obfuscated / WAF bypass
        "id=1'/**/UNION/**/SELECT/**/NULL,username,password,email/**/FROM/**/users--",
        "id=1'/**/OR/**/'1'='1'--",
        "id=1%27%20OR%201%3D1--",
        "/search?q=%27%20OR%20%271%27%3D%271%27%20UNION%20SELECT%20NULL%2Cusername%2Cpassword%20FROM%20users--",

        # Injection that contains INSERT/UPDATE/DELETE. These are attack because they break context and inject extra SQL.
        "comment=Nice post'); INSERT INTO blog_comments(comment) VALUES('stored_sqli_test');--",
        "User-Agent: Mozilla/5.0'); INSERT INTO logs(user_agent) VALUES('stored_sqli_user_agent');--",
        "name[0;UPDATE users SET name='admin' WHERE uid=1;#]=test&pass=test&form_id=user_login",
        "profile=test'; UPDATE users SET role='admin' WHERE username='attacker';--",
        "note=test'; DELETE FROM audit_logs WHERE '1'='1';--",

        # XML / SOAP-style SQLi
        "<user><name>test' OR '1'='1' --</name><id>1</id></user>",
        "<soapenv:Envelope><soapenv:Body><id>1' AND 1=1--</id></soapenv:Body></soapenv:Envelope>",

        # Other web attacks to preserve Web Attack Guard capability
        "<script>alert(1)</script>",
        "\"><script>alert(document.cookie)</script>",
        "../../../../etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "; cat /etc/passwd",
        "| whoami",
    ]


def main() -> None:
    rows: list[dict] = []

    print("[INFO] Loading truongp/web-attack-detection")
    ds = load_dataset("truongp/web-attack-detection")["train"]

    benign = ds.filter(lambda x: int(x["Label"]) == 0)
    attack = ds.filter(lambda x: int(x["Label"]) == 1)

    # Preserve original distribution knowledge.
    benign_sample = benign.shuffle(seed=SEED).select(range(min(60000, len(benign))))
    attack_sample = attack.shuffle(seed=SEED).select(range(min(60000, len(attack))))

    for row in benign_sample:
        rows.append({"text": str(row["Sentence"]), "label": 0, "source": "truongp_benign"})

    for row in attack_sample:
        rows.append({"text": str(row["Sentence"]), "label": 1, "source": "truongp_attack"})

    hard_benign = benign_sql_examples()
    hard_attacks = hard_attack_examples()

    # Add normal SQL hard negatives. Repeat to strongly reduce false positives.
    for text in hard_benign:
        rows.append({"text": text, "label": 0, "source": "hard_benign_sql"})

    for text in hard_benign:
        rows.append({"text": text, "label": 0, "source": "hard_benign_sql_repeat"})

    # Add hard attacks. Repeat to preserve SQLi/XSS/LFI detection after hard-negative tuning.
    for text in hard_attacks:
        for _ in range(35):
            rows.append({"text": text, "label": 1, "source": "hard_attack_payload"})

    random.shuffle(rows)
    write_jsonl(OUT_PATH, rows)

    label0 = sum(1 for r in rows if r["label"] == 0)
    label1 = sum(1 for r in rows if r["label"] == 1)

    print("[DONE] Wrote:", OUT_PATH)
    print("[INFO] total rows:", len(rows))
    print("[INFO] label 0 benign:", label0)
    print("[INFO] label 1 attack:", label1)


if __name__ == "__main__":
    main()
