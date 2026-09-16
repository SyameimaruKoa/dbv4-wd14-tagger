import json
import os

# スクリプトの場所を基準にする
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# レポートログはカレントディレクトリから読む（run_taggerの実行場所に依存）
REPORT_LOG_FILE = os.path.join(os.getcwd(), "report_log.json")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "config.json")


def load_config():
    """config.jsonを読み込む"""
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return None


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>WD14 Tagger Report - {category}</title>
    <style>
        body {{ background-color: #1e1e1e; color: #ddd; font-family: sans-serif; margin: 0; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #444; padding-bottom: 10px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
        .card {{ background: #2d2d2d; border-radius: 8px; overflow: hidden; padding: 10px; }}
        .img-box {{ width: 100%; height: 250px; display: flex; align-items: center; justify-content: center; background: #000; cursor: pointer; }}
        .img-box img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
        .info {{ padding-top: 10px; font-size: 12px; }}
        .bar-container {{ display: flex; align-items: center; margin-bottom: 4px; }}
        .label {{ width: 30px; font-weight: bold; }}
        .bar-bg {{ flex-grow: 1; height: 10px; background: #444; border-radius: 5px; overflow: hidden; margin: 0 5px; }}
        .bar-fill {{ height: 100%; }}
        .val {{ width: 40px; text-align: right; }}
        
        .c-gen {{ color: #4caf50; }} .bg-gen {{ background: #4caf50; }}
        .c-sen {{ color: #ffeb3b; }} .bg-sen {{ background: #ffeb3b; }}
        .c-que {{ color: #e040fb; }} .bg-que {{ background: #e040fb; }}
        .c-exp {{ color: #f44336; }} .bg-exp {{ background: #f44336; }}

        .badge {{ display: inline-block; padding: 2px 6px; font-size: 11px; font-weight: bold; border-radius: 4px; color: #fff; margin-bottom: 4px; }}
        .badge-lvl1 {{ background: #2e7d32; }}
        .badge-lvl2 {{ background: #689f38; }}
        .badge-lvl3 {{ background: #fbc02d; color: #222; }}
        .badge-lvl4 {{ background: #f57c00; }}
        .badge-lvl5 {{ background: #e64a19; }}
        .badge-lvl6 {{ background: #d32f2f; }}
        .badge-mild {{ background: #689f38; }}
        .badge-high {{ background: #e64a19; }}
        .badge-default {{ background: #555; }}
        .score-text {{ font-size: 11px; color: #aaa; margin-left: 6px; }}

        /* Modal */
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; }}
        .modal img {{ max-width: 90%; max-height: 90%; }}
    </style>
</head>
<body>
    <h1>Report: {category} ({count} images)</h1>
    <div class="grid">
        {cards}
    </div>

    <div class="modal" id="modal" onclick="this.style.display='none'">
        <img id="modal-img" src="">
    </div>

    <script>
        function show(src) {{
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'flex';
        }}
    </script>
</body>
</html>
"""

CARD_TEMPLATE = """
<div class="card">
    <div class="img-box" onclick="show('{rel_path}')">
        <img src="{rel_path}" loading="lazy" alt="img">
    </div>
    <div class="info">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
            <span class="badge {badge_cls}">{badge_text}</span>
            <span class="score-text">{score_info}</span>
        </div>
        <div class="bar-container"><span class="label c-gen">Gen</span><div class="bar-bg"><div class="bar-fill bg-gen" style="width:{p0}%"></div></div><span class="val">{p0}%</span></div>
        <div class="bar-container"><span class="label c-sen">Sen</span><div class="bar-bg"><div class="bar-fill bg-sen" style="width:{p1}%"></div></div><span class="val">{p1}%</span></div>
        <div class="bar-container"><span class="label c-que">Que</span><div class="bar-bg"><div class="bar-fill bg-que" style="width:{p2}%"></div></div><span class="val">{p2}%</span></div>
        <div class="bar-container"><span class="label c-exp">Exp</span><div class="bar-bg"><div class="bar-fill bg-exp" style="width:{p3}%"></div></div><span class="val">{p3}%</span></div>
        <div style="margin-top:5px; color:#aaa; overflow:hidden; white-space:nowrap; text-overflow:ellipsis;">{filename}</div>
    </div>
</div>
"""


def get_badge_info(rating, sen_prob):
    lvl_map = {
        "sensitive_lvl1": ("badge-lvl1", "L1"),
        "sensitive_lvl2": ("badge-lvl2", "L2"),
        "sensitive_lvl3": ("badge-lvl3", "L3"),
        "sensitive_lvl4": ("badge-lvl4", "L4"),
        "sensitive_lvl5": ("badge-lvl5", "L5"),
        "sensitive_lvl6": ("badge-lvl6", "L6"),
        "sensitive_mild": ("badge-mild", "Mild"),
        "sensitive_high": ("badge-high", "High"),
    }
    if rating in lvl_map:
        cls_name, text = lvl_map[rating]
    elif "sensitive" in rating:
        cls_name, text = "badge-lvl3", "Sen"
    elif rating == "explicit":
        cls_name, text = "badge-lvl6", "Exp"
    elif rating == "questionable":
        cls_name, text = "badge-lvl4", "Que"
    elif rating == "general":
        cls_name, text = "badge-lvl1", "Gen"
    else:
        cls_name, text = "badge-default", rating
    score_info = f"Sen: {sen_prob:.4f} ({sen_prob*100:.1f}%)"
    return cls_name, text, score_info


def make_report():
    if not os.path.exists(REPORT_LOG_FILE):
        # ログがない場合は静かに終了（呼び出し元で制御済みを想定）
        return

    config = load_config()
    folder_mapping = {}
    if config:
        folder_mapping = config.get("folder_names", {})

    try:
        with open(REPORT_LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    grouped = {}
    for item in data:
        rating = item["rating"]
        if rating not in grouped:
            grouped[rating] = []
        grouped[rating].append(item)

    generated_files = []
    for rating, items in grouped.items():
        folder_name = folder_mapping.get(rating, rating)
        html_filename = f"report_{folder_name}.html"

        cards_html = ""
        for item in items:
            abs_path = item["path"]
            try:
                rel_path = os.path.relpath(abs_path, os.getcwd())
            except ValueError:
                rel_path = "file:///" + abs_path.replace("\\", "/")

            probs = item["probs"]
            sen_prob = probs[1]
            badge_cls, badge_text, score_info = get_badge_info(rating, sen_prob)
            cards_html += CARD_TEMPLATE.format(
                rel_path=rel_path.replace("\\", "/"),
                filename=os.path.basename(abs_path),
                badge_cls=badge_cls,
                badge_text=badge_text,
                score_info=score_info,
                p0=f"{probs[0]*100:.1f}",
                p1=f"{probs[1]*100:.1f}",
                p2=f"{probs[2]*100:.1f}",
                p3=f"{probs[3]*100:.1f}",
            )

        final_html = HTML_TEMPLATE.format(
            category=folder_name, count=len(items), cards=cards_html
        )

        with open(html_filename, "w", encoding="utf-8") as f:
            f.write(final_html)

        generated_files.append(html_filename)
        print(f"[INFO] レポート作成: {html_filename}")

    # ログファイルの削除
    try:
        os.remove(REPORT_LOG_FILE)
    except OSError:
        pass


if __name__ == "__main__":
    if not os.path.exists(REPORT_LOG_FILE):
        print("[WARN] report_log.json が見つかりません。")
    else:
        make_report()
