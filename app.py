import streamlit as st
import requests

# 画面のタイトルや設定
st.set_page_config(page_title="自分風・文体変換アプリ", layout="centered")
st.title("✍️ 自分風・文体変換アプリ (Gemma 3-4B クラウド版)")

# --- 自動折り返しのためのスタイル設定 ---
st.html("""
    <style>
        code {
            white-space: pre-wrap !important;
            word-break: break-all !important;
        }
    </style>
""")

# --- APIキー（トークン）の設定 ---
if "HF_TOKEN" in st.secrets:
    hf_token = st.secrets["HF_TOKEN"]
else:
    # ローカルテスト用（ここにHugging Faceの「hf_...」を貼ればPCでもテストできます）
    hf_token = "YOUR_HF_TOKEN_HERE" 

# 使用するAIモデルをPCと同じ「Gemma 3 4B」に指定
API_URL = "https://api-inference.huggingface.co/models/google/gemma-3-4b-it"

def query_gemma3(prompt, temperature):
    """Hugging Faceの無料Inference APIを呼び出す関数"""
    if hf_token == "YOUR_HF_TOKEN_HERE" or not hf_token:
        return "エラー: Hugging FaceのAPIトークン(HF_TOKEN)が設定されていません。"
        
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Gemma 3のプロンプト構造に合わせて送信
    payload = {
        "inputs": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {
            "max_new_tokens": 1024,
            "temperature": temperature,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        res_json = response.json()
        
        # エラー処理（無料サーバーなので、少し時間が経つとスリープ状態から復帰するまで数秒待つ場合があります）
        if isinstance(res_json, dict) and "error" in res_json:
            return f"AIサーバーが準備中、または混雑しています。1分ほど待って再度お試しください: {res_json['error']}"
            
        # 返ってきたテキストを抽出
        if isinstance(res_json, list) and len(res_json) > 0:
            return res_json[0].get("generated_text", "文章の生成に失敗しました。").strip()
        return "予期せぬレスポンス形式です。"
    except Exception as e:
        return f"接続エラーが発生しました: {e}"

# --- AI処理用のプロンプト構築 ---
def analyze_style(my_sample_text):
    prompt = f"""
あなたは世界最高峰の言語アナリストです。
提示されたお手本文章を深く分析し、作者の文体の構造や執筆パターン・癖を詳細に言語化してください。

⚠️⚠️⚠️超重要・絶対遵守の禁止事項⚠️⚠️⚠️
1. 分析結果の中に、具体的な単語、名詞、概念、トピックは一切含めないでください。お手本文章から言葉を抜き出すことは絶対に禁止します。
2. カッコや鍵カッコなどの記号は一切使用しないでください。記号を使って例を出す行為も完全に禁止します。
3. あなたが報告すべきなのは、話題の中身とは無関係な、純粋な文章の組み立て方、文末のパターン、読点の打ち方という無機質な記述ルールだけです。ルールを説明する際にも具体例は作らないでください。

分析してほしいポイント：
1. 文末の癖
2. 漢字とひらがなのバランス
3. 文章のテンポと構造

余計な解説、挨拶、例示は一切不要です。分析された文章のルールだけを箇条書きで出力してください。

## お手本文章
{my_sample_text}
"""
    return query_gemma3(prompt, temperature=0.1)


def generate_by_style(ai_generated_text, style_rules):
    prompt = f"""
    あなたの役割は、文章のスタイル（文体）を変換する優秀な編集者です。
    提示された生成AIが作った元の文章の内容や主張は一切変えずに、指定された文体ルールだけを完全に適用して、自然な日本語に書き換えてください。

    【最重要指示】
    元の堅苦しい文章の表現（文末や言葉遣い）をそのまま残してはいけません。指定された文体ルールを極めて強く反映させ、あなたの言葉でしっかりと文章を再構築（リライト）してください。
    文体ルールに書かれている指示が抽象的なものであっても、元の文章の意味だけを使って表現してください。ルール側から新しい単語や概念を連想して付け足してはいけません。カッコや鍵カッコなどの記号で囲まれた例が出力に混ざることも完全に禁止します。

    余計な解説や挨拶は一切不要です。純粋に変換した文章だけを出力してください。

    ## 適用すべき文体ルール
    {style_rules}

    ## 生成AIが作った元の文章（この内容を維持して、上記の文体にハメ込んでください）
    {ai_generated_text}
    """
    return query_gemma3(prompt, temperature=0.7)


# --- 縦並びの画面レイアウト ---

# ==========================================
# セクション 1: お手本（あなたの文体）の学習
# ==========================================
st.markdown("---")
st.header("1. お手本（あなたの文体）の学習")

if 'my_sample_content' not in st.session_state:
    st.session_state['my_sample_content'] = ""
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

uploaded_files = st.file_uploader(
    "お手本テキストファイル（.txt）をアップロード:", 
    type=["txt"], 
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}"
)

if uploaded_files:
    new_texts = []
    for f in uploaded_files:
        content = f.read().decode("utf-8")
        if content not in st.session_state['my_sample_content']:
            new_texts.append(content)
    
    if new_texts:
        if st.session_state['my_sample_content']:
            st.session_state['my_sample_content'] += "\n\n" + "\n\n".join(new_texts)
        else:
            st.session_state['my_sample_content'] = "\n\n".join(new_texts)
        st.rerun()

my_sample = st.text_area(
    "現在読み込まれているお手本文章:", 
    value=st.session_state['my_sample_content'], 
    height=200, 
    placeholder="お手本となるご自身の文章をここに直接貼り付けるか、上のボタンから.txtファイルをアップロードしてください（複数追加可能）。"
)
st.session_state['my_sample_content'] = my_sample

btn_col1, btn_col2 = st.columns([1, 4])
with btn_col1:
    if st.button("🗑️ テキストクリア"):
        st.session_state['my_sample_content'] = ""
        st.session_state['uploader_key'] += 1
        if 'style_rules' in st.session_state:
            del st.session_state['style_rules']
        if 'final_result' in st.session_state:
            del st.session_state['final_result']
        st.rerun()
with btn_col2:
    if st.button("🔥 文体の特徴を分析する", type="primary"):
        if my_sample.strip() == "":
            st.error("お手本文章を入力するか、ファイルをアップロードしてください。")
        else:
            with st.spinner("AIがあなたの文体を分析中..."):
                st.session_state['style_rules'] = analyze_style(my_sample)
                st.success("分析が完了しました！下へスクロールして変換を行ってください。")

if 'style_rules' in st.session_state:
    with st.expander("💡 AIが記憶したあなたの文体ルール（クリックで開閉）", expanded=True):
        st.info(st.session_state['style_rules'])


# ==========================================
# セクション 2: 文章の文体変換
# ==========================================
st.markdown("---")
st.header("2. 文章の文体変換")

ai_generated = st.text_area(
    "生成AIに作らせた元の文章を貼り付けてください:", 
    height=150, 
    placeholder="ここに生成AIが作った堅苦しい文章などをあなた風に生成します..."
)

if st.button("✨ 自分風の文章に変換する", type="primary"):
    if 'style_rules' not in st.session_state:
        st.error("先に上のステップ1で「文体の特徴を分析」してください。")
    elif ai_generated.strip() == "":
        st.error("変換したい元の文章を入力してください。")
    else:
        with st.spinner("あなたの文体に書き換え中..."):
            final_result = generate_by_style(ai_generated, st.session_state['style_rules'])
            st.session_state['final_result'] = final_result

# 変換結果の表示エリア
if 'final_result' in st.session_state:
    st.success("🎉 変換が完了しました！")
    
    st.text_area(
        label="変換結果の内容：",
        value=st.session_state['final_result'],
        height=250,
        disabled=True,
        key="final_output_area"
    )